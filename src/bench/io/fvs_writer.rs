use crate::graph::Node;
use std::fs::OpenOptions;
use std::io::{self, Sink, Write};
use std::path::PathBuf;

/// Used to write the result DFVS of each design point of a benchmark
pub trait FvsWriter {
    fn write(&mut self, graph_label: &str, fvs: &[Node]) -> io::Result<()>;
}

/// Creates a file for every DFVS inside of a passed in directory. See [FvsFileWriter::write] for
/// the naming format of the created files.
pub struct FvsFileWriter {
    /// The directory where all DFVS files will be created
    output_dir: PathBuf,
}

impl FvsFileWriter {
    pub fn new(output_folder: PathBuf) -> Self {
        Self {
            output_dir: output_folder,
        }
    }
}

impl FvsWriter for FvsFileWriter {
    /// Writes the DFVS into a new file. Returns an error if a file with the same DFVS size for the
    /// same graph already exists.
    fn write(&mut self, graph_label: &str, fvs: &[Node]) -> io::Result<()> {
        let graph_label_enc = graph_label.replace(std::path::MAIN_SEPARATOR, "-");
        let file_name = format!("{}_k_{}.fvs", graph_label_enc, fvs.len());
        if let Ok(mut file) = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(self.output_dir.join(file_name))
        {
            write!(file, "{:?}", fvs)?;
            file.flush()?;
        }

        Ok(())
    }
}

impl FvsWriter for Sink {
    fn write(&mut self, _: &str, _: &[Node]) -> io::Result<()> {
        Ok(())
    }
}
