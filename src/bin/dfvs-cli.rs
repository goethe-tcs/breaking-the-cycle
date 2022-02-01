#![deny(warnings)]

use dfvs::graph::adj_array::AdjArrayIn;
use dfvs::graph::io::{DefaultWriter, PaceRead};
use log::info;
use std::convert::TryFrom;
use std::fs::File;
use std::io::{stdin, BufReader};
use std::path::PathBuf;
use structopt::StructOpt;

#[cfg(feature = "jemallocator")]
#[cfg(not(target_env = "msvc"))]
use jemallocator::Jemalloc;

#[cfg(feature = "jemallocator")]
#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

#[derive(Debug, StructOpt)]
#[structopt(
    name = "dfvs-cli",
    about = "Computes a directed feedback vertex set for a given input graph."
)]
struct Opt {
    /// Input file, using the graph format of the PACE 2022 challenge.
    /// `stdin` if not specified.
    #[structopt(parse(from_os_str))]
    input: Option<PathBuf>,

    /// Output file. `stdout` if not specified.
    #[structopt(parse(from_os_str))]
    output: Option<PathBuf>,

    /// Mode. 'heuristic', 'exact'. Defaults to Exact.
    /// Any invalid input fails silently to 'exact'.
    #[structopt(short, long, default_value = "exact")]
    mode: String,
}

#[derive(Debug)]
enum Mode {
    Heuristic,
    Exact,
}

impl Default for Mode {
    fn default() -> Self {
        Mode::Exact
    }
}

impl TryFrom<&str> for Mode {
    type Error = String;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "heuristic" => Ok(Mode::Heuristic),
            "exact" => Ok(Mode::Exact),
            _ => Err(format!("'{}' is an invalid Mode.", value)),
        }
    }
}

fn main() -> std::io::Result<()> {
    dfvs::log::build_pace_logger();

    let opt = Opt::from_args();
    let _mode: Mode = Mode::try_from(opt.mode.as_str()).expect("Failed parsing 'mode' flag: ");
    info!("Running in mode {:?}", _mode);

    let writer = DefaultWriter::from_path(opt.output, None)?;

    let graph: AdjArrayIn = match opt.input {
        Some(path) => {
            let file = File::open(path)?;
            AdjArrayIn::try_read_pace(BufReader::new(file))?
        }
        None => {
            let stdin = stdin();
            AdjArrayIn::try_read_pace(stdin.lock())?
        }
    };

    writer.write(&graph)?;

    Ok(())
}
