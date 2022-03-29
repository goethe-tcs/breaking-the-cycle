use std::fmt::Display;
use std::path::PathBuf;
use std::{fs, io};

pub mod bench_writer;
pub mod fvs_writer;
pub mod keyed_buffer;
pub mod keyed_csv_writer;
pub mod keyed_writer;

const BENCH_DIR_PATH: &str = "bench_output";
const EXP_LOGS_PATH: &str = "logs";

/// Returns a path to the benchmark output directory. Creates the directory if it does not exist
/// yet
pub fn bench_dir() -> io::Result<PathBuf> {
    ensure_dir(BENCH_DIR_PATH)
}

/// Returns a path to the experiment logs directory. Creates the directory if it does not exist
/// yet
pub fn logs_dir() -> io::Result<PathBuf> {
    ensure_dir(EXP_LOGS_PATH)
}

/// Returns an [`io::Error`] with the passed in message
pub fn other_io_error(msg: impl Display) -> io::Error {
    io::Error::new(io::ErrorKind::Other, msg.to_string())
}

/// Creates the directory if it doesnt exist yet and returns a [`PathBuf`] to it
fn ensure_dir(dir_path: &str) -> io::Result<PathBuf> {
    if fs::metadata(dir_path).is_err() {
        fs::create_dir(dir_path)?;
    }

    Ok(PathBuf::from(dir_path))
}
