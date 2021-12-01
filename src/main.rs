#![deny(warnings)]

use dfvs::graph::adj_array::AdjArrayIn;
use dfvs::graph::io::{PaceRead, PaceWrite};
use log::info;
use std::convert::TryFrom;
use std::fs::{File, OpenOptions};
use std::io::{stdin, stdout, BufReader};
use std::path::PathBuf;
use structopt::StructOpt;

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
    #[structopt(short, long)]
    mode: Option<String>,
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
    type Error = ();

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "heuristic" => Ok(Mode::Heuristic),
            "exact" => Ok(Mode::Exact),
            _ => Err(()),
        }
    }
}

fn main() -> std::io::Result<()> {
    #[cfg(feature = "pace-logging")]
    dfvs::log::build_pace_logger();

    let opt = Opt::from_args();
    let mode: Mode = match opt.mode {
        Some(value) => Mode::try_from(value.as_str()).unwrap_or_default(),
        None => Default::default(),
    };

    #[cfg(feature = "pace-logging")]
    info!("Running in mode {:?}", mode);

    let file = match opt.output {
        Some(path) => Some(OpenOptions::new().write(true).create(true).open(path)?),
        None => None,
    };

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
    let stdout = stdout();

    match file {
        None => {
            graph.try_write_pace(stdout.lock())?;
        }
        Some(path) => {
            graph.try_write_pace(path)?;
        }
    }

    Ok(())
}
