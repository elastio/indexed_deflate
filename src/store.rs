//! Pluggable-storage variant of the decoder and index builder.
//!
//! Where the file-backed [`GzIndexBuilder`](crate::GzIndexBuilder) /
//! [`GzDecoder`](crate::GzDecoder) read and write a single seekable index file,
//! the types in this module delegate all index persistence to a user-supplied
//! [`IndexStorage`] implementation. This unlocks three things the file-backed
//! types cannot do:
//!
//! - **Read while building, with no `finish()` call.** An index builder hands
//!   each access point to [`IndexStorage::append`] as soon as it is produced, so
//!   a separately-constructed decoder can [`find`](IndexStorage::find) and use it
//!   immediately — there is no end-of-file points table that only materializes at
//!   the end.
//! - **Any backing store.** Windows and access points can live in a database, an
//!   object store, shared memory, etc. — wherever your `IndexStorage` puts them.
//! - **Concurrency.** All [`IndexStorage`] methods take `&self`, so one handle to
//!   a shared, internally-synchronized store can be cloned into a builder on one
//!   thread and a decoder on another.
//!
//! The decompression core is shared verbatim with the file-backed types.

use std::io::{Read, Seek};

use miniz_oxide::{
    deflate::{compress_to_vec, CompressionLevel},
    inflate::{core::inflate_flags, decompress_to_vec, TINFLStatus},
};

use crate::{
    base::{Engine, ReadDecoder, SeekDecoder, SeekPoint, Wrapper, WINDOW_SIZE},
    AccessPointSpan, Error, Result,
};

/// A point in the compressed stream from which decompression can be restarted.
///
/// This is the logical record an [`IndexStorage`] backend must persist and later
/// return. It deliberately contains no storage-specific addressing (no file
/// offsets): the backend owns where the bytes live and associates the 32KB
/// [`Window`] with the point however it likes (e.g. keyed by `out_pos`).
///
/// Backends must keep points retrievable in `out_pos` order, since seeking is a
/// predecessor query (see [`IndexStorage::find`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AccessPoint {
    /// Offset in the *decompressed* stream at which this point's window begins.
    /// Access points are appended in strictly increasing `out_pos` order.
    pub out_pos: u64,

    /// Offset in the *compressed* stream at which to resume decompression.
    pub in_pos: u64,

    /// Number of leading bits of the `in_pos` byte that belong to the next
    /// DEFLATE block. When non-zero the decoder re-reads that byte on resume.
    pub num_bits: u8,
}

/// The encoding used for a stored 32KB window.
///
/// Chosen once, when an index builder is created, and applied to every window it
/// produces. Each stored [`Window`] is tagged with its format, so a decoder
/// decodes it without needing to be told the format up front.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum WindowFormat {
    /// Stored verbatim: exactly 32768 bytes. Cheapest to read back; largest at
    /// rest. Prefer this when the medium already compresses, or when you want to
    /// apply your own compression in the backend.
    Raw,

    /// DEFLATE-compressed (the same scheme the file-backed index uses, typically
    /// a few percent of the input). Cheapest at rest; costs a decompress per
    /// backward seek.
    Deflate,
}

/// A 32KB window's bytes, tagged with the [`WindowFormat`] describing them.
///
/// This is the self-describing envelope exchanged with [`IndexStorage`]: the
/// engine builds it on [`append`](IndexStorage::append) and decodes whatever the
/// backend returns from [`window`](IndexStorage::window). A backend should treat
/// the bytes as opaque and persist the format tag alongside them (e.g. a single
/// discriminant byte) so it can reconstruct the same `Window` on retrieval.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Window {
    /// Raw 32768 bytes.
    Raw(Vec<u8>),
    /// DEFLATE-compressed bytes.
    Deflate(Vec<u8>),
}

impl Window {
    /// The format of this window.
    pub fn format(&self) -> WindowFormat {
        match self {
            Window::Raw(_) => WindowFormat::Raw,
            Window::Deflate(_) => WindowFormat::Deflate,
        }
    }

    /// The stored bytes, in whatever format [`format`](Self::format) reports.
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            Window::Raw(b) | Window::Deflate(b) => b,
        }
    }

    /// Reconstruct a `Window` from a format tag and its stored bytes. Useful for
    /// backends deserializing what they previously persisted.
    pub fn from_parts(format: WindowFormat, bytes: Vec<u8>) -> Self {
        match format {
            WindowFormat::Raw => Window::Raw(bytes),
            WindowFormat::Deflate => Window::Deflate(bytes),
        }
    }
}

/// A pluggable backing store for the access points and 32KB windows that make up
/// an index.
///
/// Implement this to keep the index wherever you like (a file laid out your way,
/// a key-value store, a database, shared memory, …). One value is moved into an
/// index builder and another into a decoder; if you want to build and read the
/// same index concurrently, make `Self` a cheap, cloneable handle to a shared,
/// internally-synchronized store (all methods take `&self`).
///
/// # Consistency contract
///
/// A point must become visible to [`find`](Self::find) / [`last`](Self::last)
/// **only after** its window is durably retrievable via [`window`](Self::window).
/// Otherwise a concurrent decoder can select a point whose `window` call then
/// fails. Points are always appended in strictly increasing `out_pos` order, so a
/// monotonic "highest fully-committed `out_pos`" watermark is enough to honour
/// this.
///
/// # Errors
///
/// Any method may return `Self::Error`. The crate surfaces it as
/// [`Error::Callback`](crate::Error::Callback) from constructors / `finish`, and
/// (because `std::io::{Read, Seek}` force `io::Error`) wrapped inside an
/// `io::Error` from `read`/`seek`, where it can be recovered with
/// [`std::io::Error::get_ref`]/`downcast`.
pub trait IndexStorage {
    /// Error type returned by this backend's operations.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Persist a newly-created access point together with its window.
    ///
    /// Called by the index builder exactly once per access point, in strictly
    /// increasing `point.out_pos` order, as decompression crosses block
    /// boundaries. Honour the [consistency contract](Self#consistency-contract):
    /// do not let the point become visible to `find`/`last` until `window` can
    /// return its window.
    fn append(&self, point: &AccessPoint, window: &Window) -> std::result::Result<(), Self::Error>;

    /// Return the access point with the greatest `out_pos` that is `<= target`,
    /// or `None` if no appended point satisfies that (the decoder then restarts
    /// from the beginning of the stream).
    ///
    /// This is the predecessor query that drives seeking, so it should be cheap
    /// (e.g. a binary search / indexed lookup over `out_pos`).
    fn find(&self, target: u64) -> std::result::Result<Option<AccessPoint>, Self::Error>;

    /// Return the access point with the greatest `out_pos` appended so far, or
    /// `None` if none have been. Used to locate the tail of the index when
    /// computing the total decompressed length.
    fn last(&self) -> std::result::Result<Option<AccessPoint>, Self::Error>;

    /// Return the window previously stored alongside `point` via `append`.
    ///
    /// `point` is one that this backend returned from `find`/`last`. Return the
    /// `Window` exactly as it was appended (same format and bytes).
    fn window(&self, point: &AccessPoint) -> std::result::Result<Window, Self::Error>;

    /// Optional hook invoked by the builder's `finish()`, for backends that want
    /// to flush buffers or record a "complete" marker. The default does nothing —
    /// a decoder never requires `finish` to have been called.
    fn finalize(&self) -> std::result::Result<(), Self::Error> {
        Ok(())
    }
}

/// Encode a raw 32KB window into the chosen on-the-wire [`Window`] format.
fn encode_window(raw: Vec<u8>, format: WindowFormat) -> Window {
    match format {
        WindowFormat::Raw => Window::Raw(raw),
        WindowFormat::Deflate => {
            Window::Deflate(compress_to_vec(&raw, CompressionLevel::DefaultLevel as u8))
        }
    }
}

/// Decode a stored [`Window`] back into a raw 32KB window for the engine.
fn decode_window(window: Window) -> std::io::Result<Vec<u8>> {
    let window = match window {
        Window::Raw(bytes) => bytes,
        Window::Deflate(bytes) => decompress_to_vec(&bytes)
            .map_err(|_| std::io::Error::other("error decompressing window"))?,
    };

    if window.len() != WINDOW_SIZE as usize {
        return Err(std::io::Error::other("invalid window length"));
    }

    Ok(window)
}

/// Wrap a backend error as an `io::Error` carrying [`Error::Callback`], for use
/// inside the `Read`/`Seek` code paths (which can only return `io::Error`).
fn callback_io_err<E: std::error::Error + Send + Sync + 'static>(e: E) -> std::io::Error {
    std::io::Error::other(Error::Callback(Box::new(e)))
}

fn to_seek_point(p: AccessPoint) -> SeekPoint {
    SeekPoint {
        out_pos: p.out_pos,
        in_pos: p.in_pos,
        num_bits: p.num_bits,
    }
}

fn to_access_point(p: &SeekPoint) -> AccessPoint {
    AccessPoint {
        out_pos: p.out_pos,
        in_pos: p.in_pos,
        num_bits: p.num_bits,
    }
}

// The three seek hooks are identical for the decoder and the builder, so share them.

fn store_find<S: IndexStorage>(store: &S, target: u64) -> std::io::Result<Option<SeekPoint>> {
    Ok(store
        .find(target)
        .map_err(callback_io_err)?
        .map(to_seek_point))
}

fn store_last<S: IndexStorage>(store: &S) -> std::io::Result<Option<SeekPoint>> {
    Ok(store.last().map_err(callback_io_err)?.map(to_seek_point))
}

fn store_window<S: IndexStorage>(store: &S, point: &SeekPoint) -> std::io::Result<Vec<u8>> {
    let window = store
        .window(&to_access_point(point))
        .map_err(callback_io_err)?;
    decode_window(window)
}

/// Decoder over a [`IndexStorage`] backend. See [module docs](self).
pub(crate) struct BaseStoreDecoder<G, S> {
    engine: Engine<G>,
    store: S,
}

impl<G, S> BaseStoreDecoder<G, S>
where
    G: Read,
    S: IndexStorage,
{
    pub(crate) fn new(gz_stream: G, store: S, wrapper: Wrapper) -> Result<Self> {
        Ok(Self {
            engine: Engine::new(gz_stream, wrapper)?,
            store,
        })
    }

    pub(crate) fn header(&self) -> Option<gzip_header::GzHeader> {
        self.engine.gz_header.clone()
    }

    pub(crate) fn store(&self) -> &S {
        &self.store
    }
}

impl<G, S> ReadDecoder<G> for BaseStoreDecoder<G, S>
where
    G: Read,
    S: IndexStorage,
{
    fn engine(&mut self) -> &mut Engine<G> {
        &mut self.engine
    }

    fn make_progress(&mut self) -> std::io::Result<()> {
        let status = self.engine.make_progress(0)?;

        match status {
            TINFLStatus::Done | TINFLStatus::HasMoreOutput | TINFLStatus::NeedsMoreInput => Ok(()),
            _ => Err(std::io::Error::other("decompression failed")),
        }
    }
}

impl<G, S> SeekDecoder<G> for BaseStoreDecoder<G, S>
where
    G: Read + Seek,
    S: IndexStorage,
{
    fn find_point(&mut self, target: u64) -> std::io::Result<Option<SeekPoint>> {
        store_find(&self.store, target)
    }

    fn last_point(&mut self) -> std::io::Result<Option<SeekPoint>> {
        store_last(&self.store)
    }

    fn load_window(&mut self, point: &SeekPoint) -> std::io::Result<Vec<u8>> {
        store_window(&self.store, point)
    }
}

/// Index builder over a [`IndexStorage`] backend. See [module docs](self).
pub(crate) struct BaseStoreBuilder<G, S> {
    engine: Engine<G>,
    store: S,

    span: AccessPointSpan,
    window_format: WindowFormat,

    last_access_point: u64,
}

impl<G, S> BaseStoreBuilder<G, S>
where
    G: Read,
    S: IndexStorage,
{
    pub(crate) fn new(
        gz_stream: G,
        store: S,
        span: AccessPointSpan,
        window_format: WindowFormat,
        wrapper: Wrapper,
    ) -> Result<Self> {
        Ok(Self {
            engine: Engine::new(gz_stream, wrapper)?,
            store,
            span,
            window_format,
            last_access_point: 0,
        })
    }

    pub(crate) fn header(&self) -> Option<gzip_header::GzHeader> {
        self.engine.gz_header.clone()
    }

    pub(crate) fn store(&self) -> &S {
        &self.store
    }

    pub(crate) fn finish(self) -> Result<()> {
        self.store
            .finalize()
            .map_err(|e| Error::Callback(Box::new(e)))?;
        Ok(())
    }

    fn create_access_point(&mut self) -> std::io::Result<()> {
        let window = self.engine.current_window();

        let num_bits = self.engine.decomp.block_boundary_state().unwrap().num_bits;

        // If the next block depends on some buffered bits from the previous input
        // byte, we'll re-read that byte when resuming, to avoid having to store buf_bit
        let in_pos_offset = if num_bits == 0 { 0 } else { 1 };

        let point = AccessPoint {
            // Access point starts at the start of the window
            out_pos: self.engine.output_dec - WINDOW_SIZE,
            in_pos: self.engine.input_pos - in_pos_offset,
            num_bits,
        };

        let envelope = encode_window(window, self.window_format);

        self.store
            .append(&point, &envelope)
            .map_err(callback_io_err)?;

        Ok(())
    }
}

impl<G, S> ReadDecoder<G> for BaseStoreBuilder<G, S>
where
    G: Read,
    S: IndexStorage,
{
    fn engine(&mut self) -> &mut Engine<G> {
        &mut self.engine
    }

    fn make_progress(&mut self) -> std::io::Result<()> {
        let flags = inflate_flags::TINFL_FLAG_STOP_ON_BLOCK_BOUNDARY;
        let status = self.engine.make_progress(flags)?;

        match status {
            TINFLStatus::Done | TINFLStatus::HasMoreOutput | TINFLStatus::NeedsMoreInput => Ok(()),
            TINFLStatus::BlockBoundary => {
                if self.engine.input_pos >= self.last_access_point + self.span.0
                    && self.engine.input_pos >= WINDOW_SIZE
                {
                    self.last_access_point = self.engine.input_pos;

                    self.create_access_point()?;
                }

                Ok(())
            }
            _ => Err(std::io::Error::other("decompression failed")),
        }
    }
}

impl<G, S> SeekDecoder<G> for BaseStoreBuilder<G, S>
where
    G: Read + Seek,
    S: IndexStorage,
{
    fn find_point(&mut self, target: u64) -> std::io::Result<Option<SeekPoint>> {
        store_find(&self.store, target)
    }

    fn last_point(&mut self) -> std::io::Result<Option<SeekPoint>> {
        store_last(&self.store)
    }

    fn load_window(&mut self, point: &SeekPoint) -> std::io::Result<Vec<u8>> {
        store_window(&self.store, point)
    }
}
