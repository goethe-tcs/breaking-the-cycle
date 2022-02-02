use super::keyed_csv_writer::KeyedCsvWriter;
use super::keyed_writer::KeyedWriter;
use std::fmt::Display;
use std::io::{Result, Stdout};

/// Used to write metrics that are collected in a bench
pub trait BenchWriter {
    fn write(&mut self, column: impl Display, value: impl Display) -> Result<()>;

    fn end_design_point(&mut self) -> Result<()>;

    fn end_graph_section(&mut self) -> Result<()>;
}

pub type DualWriter = DualBenchWriter<KeyedWriter<Stdout>, KeyedCsvWriter>;
pub struct DualBenchWriter<W1, W2> {
    // can't use vector of boxed writers because the BenchWrite trait is not object safe
    // (because BenchWriter::write() has generic type parameters)
    logger_1: W1,
    logger_2: W2,
}

impl<W1: BenchWriter, W2: BenchWriter> DualBenchWriter<W1, W2> {
    pub fn new(logger1: W1, logger2: W2) -> Self {
        Self {
            logger_1: logger1,
            logger_2: logger2,
        }
    }
}

impl<W1: BenchWriter, W2: BenchWriter> BenchWriter for DualBenchWriter<W1, W2> {
    fn write(&mut self, column: impl Display, value: impl Display) -> Result<()> {
        self.logger_1.write(&column, &value)?;
        self.logger_2.write(column, value)
    }

    fn end_design_point(&mut self) -> Result<()> {
        self.logger_1.end_design_point()?;
        self.logger_2.end_design_point()
    }

    fn end_graph_section(&mut self) -> Result<()> {
        self.logger_1.end_graph_section()?;
        self.logger_2.end_graph_section()
    }
}

impl BenchWriter for () {
    fn write(&mut self, _column: impl Display, _value: impl Display) -> Result<()> {
        Ok(())
    }

    fn end_design_point(&mut self) -> Result<()> {
        Ok(())
    }

    fn end_graph_section(&mut self) -> Result<()> {
        Ok(())
    }
}
