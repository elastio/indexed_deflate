use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};

use miniz_oxide::{
    deflate::CompressionLevel,
    inflate::{
        core::{decompress, inflate_flags, BlockBoundaryState, DecompressorOxide},
        TINFLStatus,
    },
};

use crate::{AccessPointSpan, Error, Result};

/// DEFLATE max window size
pub(crate) const WINDOW_SIZE: u64 = 32768;

/// Should be large enough to get reasonably efficient reads from the input file.
const INPUT_BUF_SIZE: usize = 32768;

/// Must be at least the window size (32KB), and a power of two. Should be larger than input buffer
/// by at least the typical compression ratio, so `decompress()` can process the whole lot in one go.
pub(crate) const OUTPUT_BUF_SIZE: u64 = 65536;

/// Used for checking we're not passed a totally different format as the index file
const HEADER_MAGIC: [u8; 4] = *b"GzIx";

/// Indicates the user started writing the index file but didn't call `finish()`,
/// so it is corrupted
const VERSION_UNFINISHED: u32 = 0xffff_ffff;

/// Currently version of index format. Should be bumped whenever the format changes.
const VERSION_LATEST: u32 = 3;

/// Index file header
struct Header {
    magic: [u8; 4], // HEADER_MAGIC
    version: u32,   // VERSION_LATEST
    windows_size: u64,
}

// Index file format:
//
// struct {
//   userdata_pre: [u8; ...],
//
//   header: Header,
//   windows: [u8; header.windows_size],
//   points_count: u32,
//   points: [AccessPoint; points_count],
//
//   userdata_post: [u8; ...],
// }
//
// `windows` is a non-delimited sequence of compressed 32KB windows.
// AccessPoint contains a pointer into this region.
//
// The user can store arbitrary data before and after the index.

// Little-endian IO, to avoid a dependency on `byteorder`
fn read_u64<R: Read>(mut r: R) -> std::io::Result<u64> {
    let mut buf = [0; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_u32<R: Read>(mut r: R) -> std::io::Result<u32> {
    let mut buf = [0; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u16<R: Read>(mut r: R) -> std::io::Result<u16> {
    let mut buf = [0; 2];
    r.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

fn read_u8<R: Read>(mut r: R) -> std::io::Result<u8> {
    let mut buf = [0];
    r.read_exact(&mut buf)?;
    Ok(buf[0])
}

/// Index file header
impl Header {
    fn new(windows_size: u64) -> Self {
        Self {
            magic: HEADER_MAGIC,
            version: VERSION_LATEST,
            windows_size,
        }
    }

    fn unfinished() -> Self {
        Self {
            magic: HEADER_MAGIC,
            version: VERSION_UNFINISHED,
            windows_size: 0,
        }
    }

    fn size() -> u64 {
        16
    }

    fn read<R: Read>(mut r: R) -> std::io::Result<Self> {
        let mut magic = [0; 4];
        r.read_exact(&mut magic)?;

        let version = read_u32(&mut r)?;
        let windows_size = read_u64(&mut r)?;

        Ok(Header {
            magic,
            version,
            windows_size,
        })
    }

    fn write<W: Write>(&self, mut w: W) -> std::io::Result<()> {
        w.write_all(&self.magic)?;
        w.write_all(&self.version.to_le_bytes())?;
        w.write_all(&self.windows_size.to_le_bytes())?;
        Ok(())
    }

    fn validate(&self) -> Result<()> {
        if self.magic != HEADER_MAGIC {
            return Err(Error::IndexBadMagic);
        }
        if self.version == VERSION_UNFINISHED {
            return Err(Error::IndexUnfinished);
        }
        if self.version != VERSION_LATEST {
            return Err(Error::IndexIncompatibleVersion);
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct AccessPoint {
    out_pos: u64,
    in_pos: u64,

    /// Offset into the index file's `windows` data
    window_offset: u64,
    window_compressed_size: u16,

    /// Number of bits from the `in_pos` byte, that are part of the new deflate block
    num_bits: u8,
}

impl AccessPoint {
    fn read<R: Read>(mut r: R) -> std::io::Result<AccessPoint> {
        let out_pos = read_u64(&mut r)?;
        let in_pos = read_u64(&mut r)?;
        let window_offset = read_u64(&mut r)?;
        let window_compressed_size = read_u16(&mut r)?;
        let num_bits = read_u8(&mut r)?;

        Ok(AccessPoint {
            out_pos,
            in_pos,
            window_offset,
            window_compressed_size,
            num_bits,
        })
    }

    fn write<W: Write>(&self, mut w: W) -> std::io::Result<()> {
        w.write_all(&self.out_pos.to_le_bytes())?;
        w.write_all(&self.in_pos.to_le_bytes())?;
        w.write_all(&self.window_offset.to_le_bytes())?;
        w.write_all(&self.window_compressed_size.to_le_bytes())?;
        w.write_all(&[self.num_bits])?;
        Ok(())
    }

    fn read_all<R: Read + Seek>(r: R) -> std::io::Result<Vec<AccessPoint>> {
        let mut buf = BufReader::new(r);
        let count = read_u32(&mut buf)? as usize;
        let points: std::io::Result<_> = (0..count).map(|_| AccessPoint::read(&mut buf)).collect();

        // Sync seek position from BufReader back to R
        #[allow(clippy::seek_from_current)]
        buf.seek(SeekFrom::Current(0))?;

        points
    }

    fn write_all<W: Write>(points: &[AccessPoint], w: W) -> std::io::Result<()> {
        let mut buf = BufWriter::new(w);
        buf.write_all(&(points.len() as u32).to_le_bytes())?;
        for p in points {
            p.write(&mut buf)?;
        }
        buf.flush()?;

        Ok(())
    }
}

/// A storage-agnostic view of an access point, as understood by the shared seek
/// machinery in [`SeekDecoder`].
///
/// The on-disk file index ([`AccessPoint`]) and any pluggable storage backend both
/// project their records down to this minimal set of fields when answering seek
/// queries. The 32KB window itself is fetched separately via
/// [`SeekDecoder::load_window`], so that the forward-skip optimization can pick a
/// point without paying to load its window.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SeekPoint {
    pub out_pos: u64,
    pub in_pos: u64,
    pub num_bits: u8,
}

/// Storage-agnostic DEFLATE decompression core.
///
/// This owns the compressed input stream, the input/output buffers, and the
/// `miniz_oxide` decompressor. It knows nothing about how the index is stored,
/// so it is shared verbatim by both the file-backed types in this module and the
/// pluggable-storage types in [`crate::store`].
pub(crate) struct Engine<G> {
    pub(crate) gz_stream: G,

    pub(crate) wrapper: Wrapper,
    pub(crate) gz_header: Option<gzip_header::GzHeader>,
    pub(crate) gz_len: Option<u64>,

    /// Current gz_stream cursor, relative to cursor just after reading the gzip header.
    /// Lets us use seek_relative(), and avoids making the rest of this struct dependent
    /// on the `Seek` trait.
    pub(crate) gz_stream_pos: u64,

    pub(crate) done: bool,

    pub(crate) decomp: Box<DecompressorOxide>,

    // Input buffer, and the slice currently containing valid data
    pub(crate) input: Vec<u8>, // INPUT_BUF_SIZE
    pub(crate) input_offset: usize,
    pub(crate) input_size: usize,

    /// Position in file that corresponds to `input[input_offset]`
    pub(crate) input_pos: u64,

    /// Circular buffer for output. Size is `OUTPUT_BUF_SIZE`
    pub(crate) output: Vec<u8>,
    /// Number of bytes decoded into output buffer (not wrapped to buffer size)
    pub(crate) output_dec: u64,
    /// Number of bytes returned from output to consumer, now available for reuse
    pub(crate) output_ret: u64,
}

/// Format of the DEFLATE stream, so we can handle headers etc
pub(crate) enum Wrapper {
    Deflate,
    Gzip,
    Zlib,
}

impl<G> Engine<G>
where
    G: Read,
{
    pub(crate) fn new(mut gz_stream: G, wrapper: Wrapper) -> std::io::Result<Self> {
        let gz_header = match wrapper {
            Wrapper::Gzip => Some(gzip_header::read_gz_header(&mut gz_stream)?),
            _ => None,
        };

        Ok(Self {
            gz_stream,

            wrapper,
            gz_header,
            gz_len: None,
            gz_stream_pos: 0,

            done: false,

            decomp: Box::new(DecompressorOxide::new()),

            input: vec![0; INPUT_BUF_SIZE],
            input_offset: 0,
            input_size: 0,
            input_pos: 0,

            output: vec![0; OUTPUT_BUF_SIZE as usize],
            output_dec: 0,
            output_ret: 0,
        })
    }

    /// Try to refill the input buffer if it is empty, and return the new size.
    /// At EOF, it will return 0.
    fn refill_input(&mut self) -> std::io::Result<usize> {
        if self.input_offset >= self.input_size {
            self.input_offset = 0;
            self.input_size = self.gz_stream.read(&mut self.input)?;
            self.gz_stream_pos += self.input_size as u64;
        }

        Ok(self.input_size)
    }

    /// Read a single byte from gz_stream
    pub(crate) fn gz_read_byte(&mut self) -> std::io::Result<u8> {
        if self.refill_input()? == 0 {
            Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "read_byte failed",
            ))
        } else {
            let r = self.input[self.input_offset];
            self.input_offset += 1;
            Ok(r)
        }
    }

    pub(crate) fn make_progress(&mut self, flags: u32) -> std::io::Result<TINFLStatus> {
        // Output buffer must be fully consumed, because we're going to overwrite it
        assert!(!self.has_output());

        self.refill_input()?;

        // For Zlib format, we don't serialize the checksum (to save space),
        // so we can't check it after resuming
        let flags = flags
            | match self.wrapper {
                Wrapper::Zlib => {
                    inflate_flags::TINFL_FLAG_PARSE_ZLIB_HEADER
                        | inflate_flags::TINFL_FLAG_IGNORE_ADLER32
                }
                _ => 0,
            };

        let flags = flags
            | if self.input_size > 0 {
                inflate_flags::TINFL_FLAG_HAS_MORE_INPUT
            } else {
                0
            };

        let (status, in_consumed, out_produced) = decompress(
            &mut self.decomp,
            &self.input[self.input_offset..self.input_size],
            &mut self.output,
            (self.output_dec % OUTPUT_BUF_SIZE) as usize,
            flags,
        );

        self.input_offset += in_consumed;
        self.input_pos += in_consumed as u64;
        self.output_dec += out_produced as u64;

        if status == TINFLStatus::Done {
            self.done = true;
        }

        Ok(status)
    }
}

impl<G> Engine<G>
where
    G: Seek,
{
    /// Seek relative to just after the gzip header
    pub(crate) fn gz_seek(&mut self, from_start: u64) -> std::io::Result<()> {
        self.gz_stream
            .seek_relative(from_start as i64 - self.gz_stream_pos as i64)?;
        self.gz_stream_pos = from_start;
        Ok(())
    }
}

impl<G> Engine<G> {
    pub(crate) fn has_output(&self) -> bool {
        self.output_dec != self.output_ret
    }

    pub(crate) fn flush_output(&mut self, buf: &mut [u8]) -> usize {
        // Number of bytes to copy into buf
        let copied = ((self.output_dec - self.output_ret) as usize).min(buf.len());

        // Copy buf[i] = output[(output_ret + i) % OUTPUT_BUF_SIZE],
        // possibly split into two parts if it wraps in the output buffer

        let output_ret_idx = (self.output_ret % OUTPUT_BUF_SIZE) as usize;
        let p = copied.min(OUTPUT_BUF_SIZE as usize - output_ret_idx);
        buf[0..p].copy_from_slice(&self.output[output_ret_idx..(output_ret_idx + p)]);
        if p < copied {
            buf[p..copied].copy_from_slice(&self.output[0..(copied - p)]);
        }

        self.output_ret += copied as u64;
        copied
    }

    /// Copy a raw 32KB window into the output buffer, ending just before `output_dec`.
    pub(crate) fn install_window(&mut self, window: &[u8]) {
        assert_eq!(window.len(), WINDOW_SIZE as usize);
        for i in 0..WINDOW_SIZE {
            self.output[((self.output_dec + OUTPUT_BUF_SIZE - WINDOW_SIZE + i) % OUTPUT_BUF_SIZE)
                as usize] = window[i as usize];
        }
    }

    /// Extract the most-recently-decoded 32KB as a raw window.
    pub(crate) fn current_window(&self) -> Vec<u8> {
        (0..WINDOW_SIZE)
            .map(|i| {
                self.output[((self.output_dec + OUTPUT_BUF_SIZE - WINDOW_SIZE + i)
                    % OUTPUT_BUF_SIZE) as usize]
            })
            .collect()
    }
}

/// State shared by the two file-backed types ([`BaseDecoder`] and
/// [`BaseIndexBuilder`]): the decompression [`Engine`] plus the seekable index file.
pub(crate) struct Common<G, I> {
    pub(crate) engine: Engine<G>,
    index_stream: I,

    /// stream_position() of `Header` in `index_stream`, to support absolute seeking
    header_pos: u64,

    access_points: Vec<AccessPoint>,
}

pub(crate) struct BaseDecoder<G, I> {
    common: Common<G, I>,

    userdata_pos: u64,
}

impl<G, I> BaseDecoder<G, I>
where
    G: Read,
    I: Read + Seek,
{
    pub(crate) fn new(gz_stream: G, mut index_stream: I, wrapper: Wrapper) -> Result<Self> {
        let header_pos = index_stream.stream_position()?;

        let header = Header::read(&mut index_stream)?;
        header.validate()?;

        index_stream.seek(SeekFrom::Start(
            header_pos + Header::size() + header.windows_size,
        ))?;
        let access_points = AccessPoint::read_all(&mut index_stream)?;
        let userdata_pos = index_stream.stream_position()?;

        Ok(Self {
            common: Common {
                engine: Engine::new(gz_stream, wrapper)?,
                index_stream,
                header_pos,
                access_points,
            },
            userdata_pos,
        })
    }

    pub(crate) fn header(&self) -> Option<gzip_header::GzHeader> {
        self.common.engine.gz_header.clone()
    }

    pub(crate) fn with_index<F, T>(&mut self, f: F) -> std::io::Result<T>
    where
        F: FnOnce(&mut I) -> std::io::Result<T>,
    {
        let pos = self.common.index_stream.stream_position()?;
        self.common
            .index_stream
            .seek(SeekFrom::Start(self.userdata_pos))?;
        let r = f(&mut self.common.index_stream);
        self.common.index_stream.seek(SeekFrom::Start(pos))?;
        r
    }
}

// Shared helpers projecting the in-memory file index onto the storage-agnostic
// `SeekPoint`/window queries used by `SeekDecoder`.

fn file_find_point(access_points: &[AccessPoint], target: u64) -> Option<SeekPoint> {
    let partition = access_points.partition_point(|c| c.out_pos <= target);
    if partition == 0 {
        None
    } else {
        let p = &access_points[partition - 1];
        Some(SeekPoint {
            out_pos: p.out_pos,
            in_pos: p.in_pos,
            num_bits: p.num_bits,
        })
    }
}

fn file_last_point(access_points: &[AccessPoint]) -> Option<SeekPoint> {
    access_points.last().map(|p| SeekPoint {
        out_pos: p.out_pos,
        in_pos: p.in_pos,
        num_bits: p.num_bits,
    })
}

fn file_load_window<I: Read + Seek>(
    access_points: &[AccessPoint],
    index_stream: &mut I,
    header_pos: u64,
    point: &SeekPoint,
) -> std::io::Result<Vec<u8>> {
    // `point.out_pos` is exactly some access point's out_pos, so this recovers it.
    let partition = access_points.partition_point(|c| c.out_pos <= point.out_pos);
    let ap = &access_points[partition - 1];

    let mut window = vec![0; ap.window_compressed_size as usize];
    index_stream.seek(SeekFrom::Start(
        header_pos + Header::size() + ap.window_offset,
    ))?;
    index_stream.read_exact(&mut window)?;

    miniz_oxide::inflate::decompress_to_vec(&window)
        .map_err(|_| std::io::Error::other("error decompressing index window"))
}

impl<G, I> ReadDecoder<G> for BaseDecoder<G, I>
where
    G: Read,
    I: Read + Seek,
{
    fn engine(&mut self) -> &mut Engine<G> {
        &mut self.common.engine
    }

    fn make_progress(&mut self) -> std::io::Result<()> {
        let status = self.common.engine.make_progress(0)?;

        match status {
            TINFLStatus::Done | TINFLStatus::HasMoreOutput | TINFLStatus::NeedsMoreInput => Ok(()),
            _ => Err(std::io::Error::other("decompression failed")),
        }
    }
}

impl<G, I> SeekDecoder<G> for BaseDecoder<G, I>
where
    G: Read + Seek,
    I: Read + Seek,
{
    fn find_point(&mut self, target: u64) -> std::io::Result<Option<SeekPoint>> {
        Ok(file_find_point(&self.common.access_points, target))
    }

    fn last_point(&mut self) -> std::io::Result<Option<SeekPoint>> {
        Ok(file_last_point(&self.common.access_points))
    }

    fn load_window(&mut self, point: &SeekPoint) -> std::io::Result<Vec<u8>> {
        file_load_window(
            &self.common.access_points,
            &mut self.common.index_stream,
            self.common.header_pos,
            point,
        )
    }
}

pub(crate) struct BaseIndexBuilder<G, I> {
    common: Common<G, I>,

    span: AccessPointSpan,

    /// Size of window data in index file
    windows_size: u64,

    last_access_point: u64,
}

impl<G, I> BaseIndexBuilder<G, I>
where
    G: Read,
    I: Write + Seek,
{
    pub(crate) fn new(
        gz_stream: G,
        mut index_stream: I,
        span: AccessPointSpan,
        wrapper: Wrapper,
    ) -> std::io::Result<Self> {
        let header_pos = index_stream.stream_position()?;

        Header::unfinished().write(&mut index_stream)?;

        Ok(Self {
            common: Common {
                engine: Engine::new(gz_stream, wrapper)?,
                index_stream,
                header_pos,
                access_points: Vec::new(),
            },

            span,

            windows_size: 0,

            last_access_point: 0,
        })
    }

    pub(crate) fn header(&self) -> Option<gzip_header::GzHeader> {
        self.common.engine.gz_header.clone()
    }

    pub(crate) fn finish(mut self) -> Result<()> {
        // Serialize `access_points` onto the end of the index
        let points_pos = self.common.header_pos + Header::size() + self.windows_size;
        self.common.index_stream.seek(SeekFrom::Start(points_pos))?;
        AccessPoint::write_all(&self.common.access_points, &mut self.common.index_stream)?;

        let end_pos = self.common.index_stream.stream_position()?;

        // Construct the proper header, and overwrite the temporary one at the start of the index
        let header = Header::new(self.windows_size);
        self.common
            .index_stream
            .seek(SeekFrom::Start(self.common.header_pos))?;
        header.write(&mut self.common.index_stream)?;

        // Return to the end, so the user can append their own data
        self.common.index_stream.seek(SeekFrom::Start(end_pos))?;

        Ok(())
    }

    fn create_access_point(&mut self) -> std::io::Result<()> {
        // Copy the last 32KB from output buffer
        let window = self.common.engine.current_window();

        let window_compressed =
            miniz_oxide::deflate::compress_to_vec(&window, CompressionLevel::DefaultLevel as u8);

        // Deflate should never expand the 32KB input by 2x, so it's safe to store it as u16
        let window_compressed_size =
            u16::try_from(window_compressed.len()).expect("compressed window too large");

        let num_bits = self
            .common
            .engine
            .decomp
            .block_boundary_state()
            .unwrap()
            .num_bits;

        // If the next block depends on some buffered bits from the previous input
        // byte, we'll re-read that byte when resuming, to avoid having to store buf_bit
        let in_pos_offset = if num_bits == 0 { 0 } else { 1 };

        // Access point starts at the start of the window
        let out_pos = self.common.engine.output_dec - WINDOW_SIZE;

        // Access points must be ordered by out_pos, so we can binary-search over them
        if let Some(last) = self.common.access_points.last() {
            assert!(out_pos > last.out_pos + WINDOW_SIZE);
        }

        self.common.access_points.push(AccessPoint {
            out_pos,
            in_pos: self.common.engine.input_pos - in_pos_offset,
            window_offset: self.windows_size,
            window_compressed_size,
            num_bits,
        });
        self.common.index_stream.seek(SeekFrom::Start(
            self.common.header_pos + Header::size() + self.windows_size,
        ))?;
        self.common.index_stream.write_all(&window_compressed)?;

        self.windows_size += window_compressed_size as u64;

        Ok(())
    }
}

impl<G, I> ReadDecoder<G> for BaseIndexBuilder<G, I>
where
    G: Read,
    I: Write + Seek,
{
    fn engine(&mut self) -> &mut Engine<G> {
        &mut self.common.engine
    }

    fn make_progress(&mut self) -> std::io::Result<()> {
        let flags = inflate_flags::TINFL_FLAG_STOP_ON_BLOCK_BOUNDARY;
        let status = self.common.engine.make_progress(flags)?;

        match status {
            TINFLStatus::Done | TINFLStatus::HasMoreOutput | TINFLStatus::NeedsMoreInput => Ok(()),
            TINFLStatus::BlockBoundary => {
                if self.common.engine.input_pos >= self.last_access_point + self.span.0
                    && self.common.engine.input_pos >= WINDOW_SIZE
                {
                    self.last_access_point = self.common.engine.input_pos;

                    self.create_access_point()?;
                }

                Ok(())
            }
            _ => Err(std::io::Error::other("decompression failed")),
        }
    }
}

impl<G, I> SeekDecoder<G> for BaseIndexBuilder<G, I>
where
    G: Read + Seek,
    I: Read + Write + Seek,
{
    fn find_point(&mut self, target: u64) -> std::io::Result<Option<SeekPoint>> {
        Ok(file_find_point(&self.common.access_points, target))
    }

    fn last_point(&mut self) -> std::io::Result<Option<SeekPoint>> {
        Ok(file_last_point(&self.common.access_points))
    }

    fn load_window(&mut self, point: &SeekPoint) -> std::io::Result<Vec<u8>> {
        file_load_window(
            &self.common.access_points,
            &mut self.common.index_stream,
            self.common.header_pos,
            point,
        )
    }
}

/// Shared `Read`-side decompression behaviour.
///
/// Implementors provide access to the [`Engine`] and a `make_progress()` that
/// advances decompression (and, for index builders, records access points as
/// block boundaries are crossed). The `skip`/`read` defaults are storage-agnostic.
pub(crate) trait ReadDecoder<G>
where
    G: Read,
{
    fn engine(&mut self) -> &mut Engine<G>;

    fn make_progress(&mut self) -> std::io::Result<()>;

    fn skip(&mut self, amount: u64) -> std::io::Result<()> {
        let mut remaining = amount;
        while remaining > 0 {
            // Generate some output
            while !self.engine().has_output() && !self.engine().done {
                self.make_progress()?;
            }

            let engine = self.engine();
            let skipped = (engine.output_dec - engine.output_ret).min(remaining);
            engine.output_ret += skipped;
            remaining -= skipped;

            if skipped == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "seek past EOF",
                ));
            }
        }

        Ok(())
    }

    fn skip_to_eof(&mut self) -> std::io::Result<()> {
        loop {
            // Generate some output
            while !self.engine().has_output() && !self.engine().done {
                self.make_progress()?;
            }

            let engine = self.engine();
            let skipped = engine.output_dec - engine.output_ret;
            engine.output_ret += skipped;

            if skipped == 0 {
                return Ok(());
            }
        }
    }

    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        while !self.engine().has_output() && !self.engine().done {
            self.make_progress()?;
        }

        Ok(self.engine().flush_output(buf))
    }
}

/// Shared `Seek`-side behavior, layered on top of [`ReadDecoder`].
///
/// Implementors supply three storage hooks — [`find_point`](Self::find_point),
/// [`last_point`](Self::last_point) and [`load_window`](Self::load_window) — and
/// the `seek`/`get_gz_len` defaults drive the access-point machinery on top of
/// them. The file index answers these from its in-memory table; a pluggable
/// backend answers them from its store.
pub(crate) trait SeekDecoder<G>: ReadDecoder<G>
where
    G: Read + Seek,
{
    /// Return the access point with the greatest `out_pos <= target`, or `None`
    /// if there is no such point (in which case seeking restarts from the top).
    fn find_point(&mut self, target: u64) -> std::io::Result<Option<SeekPoint>>;

    /// Return the access point with the greatest `out_pos`, if any.
    fn last_point(&mut self) -> std::io::Result<Option<SeekPoint>>;

    /// Return the raw (decompressed) 32KB window for a point previously returned
    /// by `find_point`/`last_point`.
    fn load_window(&mut self, point: &SeekPoint) -> std::io::Result<Vec<u8>>;

    fn get_gz_len(&mut self) -> std::io::Result<u64> {
        if let Some(n) = self.engine().gz_len {
            Ok(n)
        } else {
            // Jump to the latest access point (if any)
            if let Some(last) = self.last_point()? {
                self.seek(SeekFrom::Start(last.out_pos))?;
            }

            // Decode up to EOF
            self.skip_to_eof()?;

            let n = self.engine().output_ret;
            self.engine().gz_len = Some(n);
            Ok(n)
        }
    }

    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        let target = match pos {
            SeekFrom::Start(n) => n,
            SeekFrom::End(n) => (self.get_gz_len()? as i64 + n) as u64,
            SeekFrom::Current(n) => (self.engine().output_ret as i64 + n) as u64,
        };

        // If the target is already inside output_buf, we can jump there directly
        {
            let engine = self.engine();
            if target >= engine.output_ret && target <= engine.output_dec {
                engine.output_ret = target;
                return Ok(target);
            }
        }

        let found = self.find_point(target)?;
        let has_point = found.is_some();
        let point = found.unwrap_or(SeekPoint {
            out_pos: 0,
            in_pos: 0,
            num_bits: 0,
        });

        // If we're seeking forwards, and we're already after the chosen access point, we shouldn't
        // go backwards to the access point. Even if we're ~32KB before the access point, it's
        // probably cheaper to keep decompressing from here instead of reloading the window data
        let fast_skip = {
            let engine = self.engine();
            if target >= engine.output_ret && engine.output_ret + WINDOW_SIZE >= point.out_pos {
                Some(target - engine.output_ret)
            } else {
                None
            }
        };
        if let Some(skip) = fast_skip {
            self.skip(skip)?;
            return Ok(target);
        }

        // Load the window before reconfiguring the engine, to avoid overlapping borrows
        // (and to skip the load entirely when restarting from the top).
        let window = if has_point {
            Some(self.load_window(&point)?)
        } else {
            None
        };

        let engine = self.engine();
        engine.gz_seek(point.in_pos)?;
        engine.input_offset = 0;
        engine.input_size = 0;
        engine.input_pos = point.in_pos;
        engine.output_ret = point.out_pos;
        engine.done = false;

        if let Some(window) = window {
            // Output buffer contains the window data
            engine.output_dec = point.out_pos + WINDOW_SIZE;

            // If the block was not byte-aligned, extract the bits from the first byte
            let bit_buf = if point.num_bits != 0 {
                engine.gz_read_byte()? >> (8 - point.num_bits)
            } else {
                0
            };

            *engine.decomp = DecompressorOxide::from_block_boundary_state(&BlockBoundaryState {
                num_bits: point.num_bits,
                bit_buf,
                ..Default::default()
            });

            engine.install_window(&window);
        } else {
            // Output buffer is empty
            engine.output_dec = point.out_pos;

            // Restart from the start of the file
            *engine.decomp = DecompressorOxide::new();
        }

        self.skip(target - point.out_pos)?;
        Ok(target)
    }
}
