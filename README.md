# `indexed_deflate`

Gzip/Zlib/DEFLATE decoder with efficient random access.

As DEFLATE does not normally support random access, we build an index while decompressing the
entire input. This contains a set of access points, typically one per 1MB of input.
We can restart decompression from any access point, letting us seek to any byte for the
cost of decompressing at most 1MB of discarded data (a few milliseconds on a desktop CPU).

The index can be reused for any subsequent processing of the same file. It is stored either in
a single seekable index file (the file-based API) or in an arbitrary backing store via a
pluggable `IndexStorage` trait (see [Pluggable storage](#pluggable-storage)).

Decompression is implemented with the pure-Rust [`miniz_oxide`](https://crates.io/crates/miniz_oxide).

## Performance

With the default configuration, the index file stored on disk will be up to 3% of the size of the input file.
Only a small map of file offsets is stored in RAM, roughly 0.003% of the size of the input.

This minimises the startup cost when a process only wants to use a small part of the index:
the total time to open, seek and start reading is only a few milliseconds, even if the input
file is many GBs.

## Usage (file-based index)

An example implementing random access to `.tar.gz` files:

```rust
use std::{collections::HashMap, fs::File, io::{Read, Seek, SeekFrom, Write}, str};
use indexed_deflate::{AccessPointSpan, GzDecoder, GzIndexBuilder, Result};

fn build_tar_index() -> Result<()> {
    let gz = File::open("example.tar.gz")?;
    let mut index = File::create("example.tar.gz.index")?;

    // GzIndexBuilder supports Read and Seek
    let mut builder = GzIndexBuilder::new(gz, &index, AccessPointSpan::default())?;

    // Extract the tar file listing, while decompressing
    let mut archive = tar::Archive::new(&mut builder);
    let files: HashMap<String, (u64, u64)> = archive
        .entries_with_seek()?
        .map(|file| {
            let file = file.unwrap();
            let path = str::from_utf8(&file.path_bytes()).unwrap().to_owned();
            (path, (file.raw_file_position(), file.size()))
        })
        .collect();

    // Finish writing the index to disk
    builder.finish()?;

    // Append our serialized file listing to the index file
    index.write_all(&postcard::to_stdvec(&files).unwrap())?;

    Ok(())
}

fn use_tar_index() -> Result<()> {
    let gz = File::open("example.tar.gz")?;
    let index = File::open("example.tar.gz.index")?;

    // GzDecoder supports Read and Seek
    let mut stream = GzDecoder::new(gz, index)?;

    // Load the tar file listing from the end of the index file
    let files: HashMap<String, (u64, u64)> = stream.with_index(|index| {
        let mut buf = Vec::new();
        index.read_to_end(&mut buf)?;
        Ok(postcard::from_bytes(&buf).unwrap())
    })?;

    let (file_pos, file_size) = files.get("example.txt").unwrap();

    // Seek in the decompressed stream to read the file
    stream.seek(SeekFrom::Start(*file_pos))?;
    let mut buf = vec![0; *file_size as usize];
    stream.read_exact(&mut buf)?;

    println!("{}", str::from_utf8(&buf).unwrap());

    Ok(())
}
```

## Pluggable storage

Instead of a single index file, access points and their 32KB windows can be stored in any
backend by implementing the `IndexStorage` trait. The store-backed types — `GzStoreIndexBuilder`
/ `GzStoreDecoder` (and the `Deflate*` / `Zlib*` variants) — use it in place of the index stream.

This unlocks behaviours the file-based API cannot offer:

- **Read while building, with no `finish()` call.** Each access point is handed to
  `IndexStorage::append` as soon as it is produced, so a separately-constructed decoder can find
  and use it immediately — there is no end-of-file points table that only materialises at the end.
- **Any backing store.** Windows and access points can live in a database, an object store,
  shared memory, etc. — wherever the `IndexStorage` implementation puts them.
- **Concurrency.** Every `IndexStorage` method takes `&self`, so one cheaply-cloneable handle to a
  shared, internally-synchronised store can feed a builder on one thread and a decoder on another.

The window encoding is chosen when the builder is created, via `WindowFormat`: `Deflate`
(compact, like the file-based index) or `Raw` (32KB verbatim, cheapest to read back — useful when
the medium already compresses, or when the backend applies its own compression). Each stored
window is self-describing, so the decoder needs no format argument.

The trait to implement:

```rust
use indexed_deflate::{AccessPoint, IndexStorage, Window};

pub trait IndexStorage {
    type Error: std::error::Error + Send + Sync + 'static;

    /// Persist a new access point + its window. Called in strictly increasing
    /// `point.out_pos` order. A point must not become visible to `find`/`last`
    /// until its window is retrievable via `window`.
    fn append(&self, point: &AccessPoint, window: &Window) -> Result<(), Self::Error>;

    /// The access point with the greatest `out_pos <= target`, if any
    /// (the predecessor query that drives seeking).
    fn find(&self, target: u64) -> Result<Option<AccessPoint>, Self::Error>;

    /// The access point with the greatest `out_pos`, if any.
    fn last(&self) -> Result<Option<AccessPoint>, Self::Error>;

    /// The window previously stored alongside `point`.
    fn window(&self, point: &AccessPoint) -> Result<Window, Self::Error>;

    /// Optional completion hook; the default does nothing.
    fn finalize(&self) -> Result<(), Self::Error> { Ok(()) }
}
```

Building and reading, sharing one store (here a thread-safe in-memory map) — note no `finish()` is
required before the decoder can read:

```rust
use std::{fs::File, io::{Read, Seek, SeekFrom}};
use indexed_deflate::{AccessPointSpan, GzStoreDecoder, GzStoreIndexBuilder, Result, WindowFormat};

fn build_and_use<S: indexed_deflate::IndexStorage + Clone>(store: S) -> Result<()> {
    // Build the index into `store`, while decompressing the whole file.
    let mut builder = GzStoreIndexBuilder::new(
        File::open("example.gz")?,
        store.clone(),
        AccessPointSpan::default(),
        WindowFormat::Deflate,
    )?;
    builder.seek(SeekFrom::End(0))?;

    // A decoder sharing the same store can seek immediately — even concurrently,
    // and even before the builder has finished.
    let mut decoder = GzStoreDecoder::new(File::open("example.gz")?, store)?;
    decoder.seek(SeekFrom::Start(100_000_000))?;
    let mut buf = vec![0u8; 1024];
    decoder.read_exact(&mut buf)?;

    Ok(())
}
```

Unlike the file-based index there is no shared index file to append a file listing to, so the
caller owns any such metadata (e.g. the `.tar.gz` file map is kept in the backing store or a
sidecar). Also note `SeekFrom::End` decodes to EOF, so it assumes the full compressed stream is
present.
