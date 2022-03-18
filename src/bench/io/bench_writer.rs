use super::keyed_csv_writer::KeyedCsvWriter;
use super::keyed_writer::KeyedWriter;
use crate::bench::io::keyed_buffer::KeyedBuffer;
use std::fmt::Display;
use std::io::{self, Sink, Stdout};

/// Used to write metrics that are collected in a benchmark
pub trait BenchWriter {
    fn write(&mut self, key: impl Display, value: impl Display) -> io::Result<()>;

    fn end_design_point(&mut self) -> io::Result<()>;

    fn end_bench(&mut self) -> io::Result<()>;

    fn add_buffer_content(&mut self, buffer: &mut KeyedBuffer) -> io::Result<()> {
        let values = buffer.flush();
        for (key, value) in buffer.get_keys().iter().zip(values) {
            self.write(key, value)?;
        }
        Ok(())
    }
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
    fn write(&mut self, key: impl Display, value: impl Display) -> io::Result<()> {
        self.logger_1
            .write(&key, &value)
            .and(self.logger_2.write(&key, &value))
    }

    fn end_design_point(&mut self) -> io::Result<()> {
        self.logger_1
            .end_design_point()
            .and(self.logger_2.end_design_point())
    }

    fn end_bench(&mut self) -> io::Result<()> {
        self.logger_1.end_bench().and(self.logger_2.end_bench())
    }
}

impl BenchWriter for Sink {
    fn write(&mut self, _key: impl Display, _value: impl Display) -> io::Result<()> {
        Ok(())
    }

    fn end_design_point(&mut self) -> io::Result<()> {
        Ok(())
    }

    fn end_bench(&mut self) -> io::Result<()> {
        Ok(())
    }
}
