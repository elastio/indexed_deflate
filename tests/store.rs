use std::{
    collections::HashMap,
    io::{Read, Seek, SeekFrom, Write},
    sync::{Arc, Mutex},
};

use indexed_deflate::{
    AccessPoint, AccessPointSpan, GzStoreDecoder, GzStoreIndexBuilder, IndexStorage, Result, Window,
    WindowFormat,
};
use rand::{seq::SliceRandom, RngCore, SeedableRng};
use sha2::{Digest, Sha256};

fn data_random(i: u64) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update(i.to_le_bytes());
    hasher.finalize().to_vec()
}

#[derive(Debug)]
struct MemError(&'static str);

impl std::fmt::Display for MemError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MemError: {}", self.0)
    }
}

impl std::error::Error for MemError {}

/// A trivial in-memory, thread-safe, cheaply-cloneable index store.
/// Two clones share the same `Vec` via `Arc<Mutex<_>>`, so one can be moved into
/// a builder and another into a decoder.
#[derive(Clone, Default)]
struct MemStore {
    points: Arc<Mutex<Vec<(AccessPoint, Window)>>>,
}

impl IndexStorage for MemStore {
    type Error = MemError;

    fn append(&self, point: &AccessPoint, window: &Window) -> std::result::Result<(), MemError> {
        let mut points = self.points.lock().unwrap();
        if let Some((last, _)) = points.last() {
            assert!(point.out_pos > last.out_pos, "points must be appended in order");
        }
        points.push((*point, window.clone()));
        Ok(())
    }

    fn find(&self, target: u64) -> std::result::Result<Option<AccessPoint>, MemError> {
        let points = self.points.lock().unwrap();
        // Appended in increasing out_pos order, so the predecessor is the last
        // element with out_pos <= target.
        Ok(points
            .iter()
            .rev()
            .find(|(p, _)| p.out_pos <= target)
            .map(|(p, _)| *p))
    }

    fn last(&self) -> std::result::Result<Option<AccessPoint>, MemError> {
        let points = self.points.lock().unwrap();
        Ok(points.last().map(|(p, _)| *p))
    }

    fn window(&self, point: &AccessPoint) -> std::result::Result<Window, MemError> {
        let points = self.points.lock().unwrap();
        points
            .iter()
            .find(|(p, _)| p.out_pos == point.out_pos)
            .map(|(_, w)| w.clone())
            .ok_or(MemError("window not found"))
    }
}

fn make_gz(num_chunks: u64) -> tempfile::NamedTempFile {
    let gz = tempfile::NamedTempFile::new().unwrap();
    let mut encoder = flate2::write::GzEncoder::new(&gz, flate2::Compression::default());
    for i in 0..num_chunks {
        encoder.write_all(&data_random(i)).unwrap();
    }
    encoder.finish().unwrap();
    gz
}

/// Build the whole index into a store (never calling `finish()`), then random-seek
/// with a separately-constructed decoder sharing that store.
fn build_then_decode(window_format: WindowFormat) -> Result<()> {
    let chunk_size = Sha256::output_size() as u64;
    let num_chunks = (4 * 1024 * 1024) / chunk_size;

    let gz = make_gz(num_chunks);
    let store = MemStore::default();

    {
        let mut builder = GzStoreIndexBuilder::new(
            gz.reopen()?,
            store.clone(),
            AccessPointSpan::new(128 * 1024),
            window_format,
        )?;
        builder.seek(SeekFrom::End(0))?;
        // Deliberately no finish() — the decoder must work without it.
    }

    assert!(!store.points.lock().unwrap().is_empty());

    let mut decoder = GzStoreDecoder::new(gz.reopen()?, store)?;

    let mut rng = rand_pcg::Pcg64::seed_from_u64(1);
    let mut chunks: Vec<_> = (0..num_chunks).collect();
    chunks.shuffle(&mut rng);
    for &c in &chunks[0..1024] {
        decoder.seek(SeekFrom::Start(c * chunk_size))?;
        let mut buf = vec![0; chunk_size as usize];
        decoder.read_exact(&mut buf)?;
        assert_eq!(buf, data_random(c));
    }

    Ok(())
}

#[test]
fn store_build_then_decode_deflate() -> Result<()> {
    build_then_decode(WindowFormat::Deflate)
}

#[test]
fn store_build_then_decode_raw() -> Result<()> {
    build_then_decode(WindowFormat::Raw)
}

/// The headline capability: a separate decoder reads access points from the store
/// *while the builder is only partway through* building the index.
#[test]
fn store_read_while_building() -> Result<()> {
    let chunk_size = Sha256::output_size() as u64;
    let num_chunks = (4 * 1024 * 1024) / chunk_size;

    let gz = make_gz(num_chunks);
    let store = MemStore::default();

    let mut builder = GzStoreIndexBuilder::new(
        gz.reopen()?,
        store.clone(),
        AccessPointSpan::new(128 * 1024),
        WindowFormat::Deflate,
    )?;

    // Build the first half by reading sequentially; this appends access points
    // covering the first ~2MB of decompressed output.
    let half = num_chunks / 2;
    let mut buf = vec![0; chunk_size as usize];
    for i in 0..half {
        builder.read_exact(&mut buf)?;
        assert_eq!(buf, data_random(i));
    }
    let points_so_far = store.points.lock().unwrap().len();
    assert!(points_so_far > 1, "expected several access points to exist mid-build");

    // A SEPARATE decoder, sharing the store, seeks into the already-built region.
    {
        let mut decoder = GzStoreDecoder::new(gz.reopen()?, store.clone())?;
        let mut rng = rand_pcg::Pcg64::seed_from_u64(2);
        let mut chunks: Vec<_> = (0..half).collect();
        chunks.shuffle(&mut rng);
        for &c in &chunks[0..256] {
            decoder.seek(SeekFrom::Start(c * chunk_size))?;
            let mut b = vec![0; chunk_size as usize];
            decoder.read_exact(&mut b)?;
            assert_eq!(b, data_random(c));
        }
    }

    // Finish building, then the same decoder kind can reach the back half too.
    builder.seek(SeekFrom::End(0))?;
    assert!(store.points.lock().unwrap().len() > points_so_far);

    let mut decoder = GzStoreDecoder::new(gz.reopen()?, store)?;
    decoder.seek(SeekFrom::Start((num_chunks - 1) * chunk_size))?;
    let mut b = vec![0; chunk_size as usize];
    decoder.read_exact(&mut b)?;
    assert_eq!(b, data_random(num_chunks - 1));

    Ok(())
}

/// Deterministic pseudo-random file contents of a given length.
fn file_content(seed: u64, len: usize) -> Vec<u8> {
    let mut rng = rand_pcg::Pcg64::seed_from_u64(seed);
    let mut v = vec![0u8; len];
    rng.fill_bytes(&mut v);
    v
}

/// Real-world scenario: build a `.tar.gz`, index it with the store-backed builder
/// while extracting the tar file listing, then randomly access individual entries
/// through a decoder sharing that store and verify their contents.
#[test]
fn store_targz_random_access() -> Result<()> {
    // A spread of file sizes: some smaller than the access-point span, some much
    // larger (so several access points land inside a single file).
    let files: Vec<(String, Vec<u8>)> = [
        ("readme.txt", 13usize),
        ("small.bin", 5_000),
        ("dir/a.dat", 100_000),
        ("dir/b.dat", 700_000),
        ("tiny", 32),
        ("dir/nested/c.dat", 250_000),
        ("big.bin", 1_000_000),
        ("dir/d.dat", 64_000),
        ("e.dat", 400_000),
        ("last.bin", 900_000),
    ]
    .iter()
    .enumerate()
    .map(|(i, (name, len))| (name.to_string(), file_content(i as u64, *len)))
    .collect();

    // Build an uncompressed tar in memory.
    let mut tar_builder = tar::Builder::new(Vec::new());
    for (name, content) in &files {
        let mut header = tar::Header::new_gnu();
        header.set_size(content.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        tar_builder.append_data(&mut header, name, &content[..])?;
    }
    let tar_bytes = tar_builder.into_inner()?;

    // Gzip it into a temp file.
    let gz = tempfile::NamedTempFile::new()?;
    let mut encoder = flate2::write::GzEncoder::new(&gz, flate2::Compression::default());
    encoder.write_all(&tar_bytes)?;
    encoder.finish()?;

    let store = MemStore::default();

    // Index the tarball, extracting each entry's offset/size from the tar headers
    // as we decompress. `entries_with_seek` requires the builder to be Read + Seek.
    let listing: HashMap<String, (u64, u64)> = {
        let mut builder = GzStoreIndexBuilder::new(
            gz.reopen()?,
            store.clone(),
            AccessPointSpan::default(),
            WindowFormat::Deflate,
        )?;

        let listing = {
            let mut archive = tar::Archive::new(&mut builder);
            archive
                .entries_with_seek()?
                .map(|entry| {
                    let entry = entry.unwrap();
                    let path = entry.path().unwrap().to_string_lossy().into_owned();
                    (path, (entry.raw_file_position(), entry.size()))
                })
                .collect()
        };

        // Optional for the store path, but representative of the normal flow.
        builder.finish()?;
        listing
    };

    assert_eq!(listing.len(), files.len());
    assert!(
        !store.points.lock().unwrap().is_empty(),
        "indexing the tarball should have produced access points"
    );

    // Random access: a fresh decoder sharing the store seeks straight to each
    // entry (in shuffled order) and reads back exactly its bytes.
    let mut decoder = GzStoreDecoder::new(gz.reopen()?, store)?;

    let mut order: Vec<usize> = (0..files.len()).collect();
    order.shuffle(&mut rand_pcg::Pcg64::seed_from_u64(42));

    for &i in &order {
        let (name, content) = &files[i];
        let (pos, size) = listing[name];
        assert_eq!(size, content.len() as u64);

        decoder.seek(SeekFrom::Start(pos))?;
        let mut buf = vec![0; size as usize];
        decoder.read_exact(&mut buf)?;
        assert_eq!(&buf, content, "contents of {name} did not match");
    }

    Ok(())
}

/// The builder itself can seek backwards using the partially-built store.
#[test]
fn store_builder_self_seek() -> Result<()> {
    let chunk_size = Sha256::output_size() as u64;
    let num_chunks = (2 * 1024 * 1024) / chunk_size;

    let gz = make_gz(num_chunks);
    let store = MemStore::default();

    let mut builder = GzStoreIndexBuilder::new(
        gz.reopen()?,
        store,
        AccessPointSpan::new(128 * 1024),
        WindowFormat::Raw,
    )?;

    builder.seek(SeekFrom::End(0))?;

    let mut rng = rand_pcg::Pcg64::seed_from_u64(3);
    let mut chunks: Vec<_> = (0..num_chunks).collect();
    chunks.shuffle(&mut rng);
    for &c in &chunks[0..512] {
        builder.seek(SeekFrom::Start(c * chunk_size))?;
        let mut b = vec![0; chunk_size as usize];
        builder.read_exact(&mut b)?;
        assert_eq!(b, data_random(c));
    }

    Ok(())
}
