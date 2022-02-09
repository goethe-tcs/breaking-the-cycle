use std::fmt::Display;
use std::path::PathBuf;
use std::{fs, io};

pub mod bench_writer;
pub mod design_point_buffer;
pub mod fvs_writer;
pub mod keyed_buffer;
pub mod keyed_csv_writer;
pub mod keyed_writer;

const BENCH_DIR_PATH: &str = "bench_output";

/// Returns a path to the benchmark output directory. Creates the directory if it does not exist
/// yet
pub fn bench_dir() -> std::io::Result<PathBuf> {
    if fs::metadata(BENCH_DIR_PATH).is_err() {
        fs::create_dir(BENCH_DIR_PATH)?;
    }

    Ok(PathBuf::from(BENCH_DIR_PATH))
}

pub fn other_io_error<T>(msg: impl Display) -> io::Result<T> {
    Err(io::Error::new(io::ErrorKind::Other, msg.to_string()))
}
