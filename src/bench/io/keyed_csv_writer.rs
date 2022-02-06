use super::bench_writer::BenchWriter;
use super::keyed_buffer::KeyedBuffer;
use csv::{QuoteStyle, ReaderBuilder, Terminator, Writer, WriterBuilder};
use std::fmt::Display;
use std::fs::File;
use std::io::Result;
use std::path::{Path, PathBuf};

/// A wrapper around `csv::Writer<File>` that:
///  - doesn't require consistent order of entry insertion
///  - nor consistent amount of entries per line
///  - and supports adding columns on the run
///
/// If new columns were added after writing the first line, `fix_columns()` has to be called to
/// ensures that all column labels are written to the file and already written lines are filled
/// with empty entries for the new columns.
///
/// `fix_columns()` is called automatically in the destructor.
pub struct KeyedCsvWriter {
    line_buffer: KeyedBuffer,
    is_first_line: bool,

    file_path: PathBuf,
    writer: Writer<File>,
    builder: KeyedCsvWriterBuilder,
}

impl KeyedCsvWriter {
    /// Creates a new writer that creates/overwrites a file at the given path.
    pub fn new(file_path: PathBuf) -> Result<Self> {
        Self::from_builder(file_path, KeyedCsvWriterBuilder::default())
    }

    pub fn from_builder(file_path: PathBuf, builder: KeyedCsvWriterBuilder) -> Result<Self> {
        let writer_builder: WriterBuilder = builder.clone().into();
        let writer = writer_builder.from_path(file_path.as_path())?;

        Ok(Self {
            line_buffer: KeyedBuffer::new(),
            is_first_line: true,
            file_path,
            writer,
            builder,
        })
    }

    /// Write one entry into the line buffer, new columns are added automatically.
    pub fn write(&mut self, column: impl Display, value: impl Display) {
        self.line_buffer.write(column, value);
    }

    /// Ends the current line and writes it to the file
    pub fn end_line(&mut self) -> Result<()> {
        // write header
        if self.is_first_line {
            self.is_first_line = false;
            self.writer.write_record(self.line_buffer.get_columns())?;
        }

        // write line
        let csv_row = self.line_buffer.flush();
        self.writer.write_record(csv_row)?;
        self.writer.flush()?;

        Ok(())
    }

    /// Rewrites the whole file if columns where added after the first line was written. This will
    /// ensure that all column headers are included in the file and that incomplete lines have an
    /// empty entry for the new columns.
    pub fn fix_columns(&mut self) -> Result<()> {
        let columns = self.line_buffer.get_columns();

        // create reader
        let mut builder: ReaderBuilder = self.builder.clone().into();
        builder.flexible(true);
        let mut reader = builder.from_path(self.file_path.clone())?;

        // skip if no new columns where added after the first line was written
        // length between lines will be consistent if this is the case
        let headers = reader.headers()?;
        if headers.len() == columns.len() {
            return Ok(());
        }

        let records: std::result::Result<Vec<_>, _> = reader.records().collect();
        let writer_builder: WriterBuilder = self.builder.clone().into();
        self.writer = writer_builder.from_path(self.file_path.clone())?;

        // rewrite file
        self.writer.write_record(columns)?;
        for record in records? {
            let mut row = Vec::with_capacity(columns.len());
            row.extend(record.iter());

            // add missing (empty) entries to new columns
            debug_assert!(row.len() <= columns.len());
            if row.len() < columns.len() {
                row.resize(columns.len(), "");
            }

            self.writer.write_record(row)?;
        }

        self.writer.flush()?;
        Ok(())
    }

    pub fn get_file_path(&self) -> &Path {
        self.file_path.as_path()
    }
}

impl BenchWriter for KeyedCsvWriter {
    fn write(&mut self, column: impl Display, value: impl Display) -> Result<()> {
        self.write(column, value);
        Ok(())
    }

    fn end_design_point(&mut self) -> Result<()> {
        self.end_line()
    }

    fn end_graph_section(&mut self) -> Result<()> {
        self.fix_columns()
    }
}

impl Drop for KeyedCsvWriter {
    /// end the current line and fix columns in destructor
    fn drop(&mut self) {
        if !self.line_buffer.is_empty() {
            self.end_line().expect("Failed to flush line in destructor");
        }

        self.fix_columns()
            .expect("Failed to fix columns in destructor");
    }
}

/// Used to store values for both csv::WriterBuilder and csv::ReaderBuilder
#[derive(Clone)]
pub struct KeyedCsvWriterBuilder {
    quote: u8,
    double_quote: bool,
    quote_style: QuoteStyle,

    terminator: Terminator,
    flexible: bool,
    delimiter: u8,
    escape: u8,
}

impl KeyedCsvWriterBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn quote(&mut self, new_quote: u8) {
        self.quote = new_quote;
    }

    pub fn double_quote(&mut self, new_double_quote: bool) {
        self.double_quote = new_double_quote;
    }

    pub fn quote_style(&mut self, new_quote_style: QuoteStyle) {
        self.quote_style = new_quote_style;
    }

    pub fn terminator(&mut self, new_terminator: Terminator) {
        self.terminator = new_terminator;
    }

    pub fn flexible(&mut self, new_flexible: bool) {
        self.flexible = new_flexible;
    }

    pub fn delimiter(&mut self, new_delimiter: u8) {
        self.delimiter = new_delimiter;
    }

    pub fn escape(&mut self, new_escape: u8) {
        self.escape = new_escape;
    }
}

impl Default for KeyedCsvWriterBuilder {
    /// Creates a new builder with default values which were copied from here:
    /// https://github.com/BurntSushi/rust-csv/blob/40ea4c49d7467d2b607a6396424f8e0e101adae1/csv-core/src/writer.rs#L20
    fn default() -> Self {
        Self {
            quote: b'"',
            double_quote: true,
            quote_style: QuoteStyle::default(),

            terminator: Terminator::Any(b'\n'),
            flexible: true,
            delimiter: b',',
            escape: b'\\',
        }
    }
}

impl From<KeyedCsvWriterBuilder> for WriterBuilder {
    fn from(other: KeyedCsvWriterBuilder) -> Self {
        let mut builder = WriterBuilder::default();
        builder.quote(other.quote);
        builder.double_quote(other.double_quote);
        builder.quote_style(other.quote_style);

        builder.terminator(other.terminator);
        builder.flexible(other.flexible);
        builder.delimiter(other.delimiter);
        builder.escape(other.escape);

        builder
    }
}

impl From<KeyedCsvWriterBuilder> for ReaderBuilder {
    fn from(other: KeyedCsvWriterBuilder) -> Self {
        let mut builder = ReaderBuilder::default();
        builder.quote(other.quote);
        builder.double_quote(other.double_quote);

        builder.terminator(other.terminator);
        builder.flexible(other.flexible);
        builder.delimiter(other.delimiter);
        builder.escape(Some(other.escape));

        builder
    }
}

#[cfg(feature = "tempfile")]
#[cfg(test)]
mod tests_keyed_csv_writer {
    use super::*;
    use csv::Reader;
    use std::result::Result;

    #[test]
    fn test_inconsistent_writes() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.csv");

        // test destructor
        {
            let mut writer = KeyedCsvWriter::new(file_path.clone()).unwrap();
            writer.write("column 1", "entry 1 1");
            writer.write("column 2", "entry 1 2");
            writer.write("column 3", "entry 1 3");
            writer.end_line().unwrap();

            writer.write("column 3", "entry 2 3");
            writer.write("column 1", "entry 2 1");
        }

        let mut reader = Reader::from_path(file_path).unwrap();
        let headers = reader.headers().unwrap();
        assert_eq!(headers, vec!["column 1", "column 2", "column 3"]);

        let records: Result<Vec<_>, _> = reader.records().collect();
        assert_eq!(
            records.unwrap(),
            vec![
                vec!["entry 1 1", "entry 1 2", "entry 1 3"],
                vec!["entry 2 1", "", "entry 2 3"]
            ]
        );
    }

    #[test]
    fn test_fix_columns() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.csv");

        // test destructor
        {
            let mut writer = KeyedCsvWriter::new(file_path.clone()).unwrap();
            writer.write("column 1", "entry 1 1");
            writer.write("column 2", "entry 1 2");
            writer.write("column 3", "entry 1 3");
            writer.end_line().unwrap();

            writer.write("column 4", "entry 2 4");
            writer.write("column 3", "entry 2 3");
            writer.write("column 1", "entry 2 1");
        }

        let mut reader = Reader::from_path(file_path).unwrap();
        let headers = reader.headers().unwrap();
        assert_eq!(
            headers,
            vec!["column 1", "column 2", "column 3", "column 4"]
        );

        let records: Result<Vec<_>, _> = reader.records().collect();
        assert_eq!(
            records.unwrap(),
            vec![
                vec!["entry 1 1", "entry 1 2", "entry 1 3", ""],
                vec!["entry 2 1", "", "entry 2 3", "entry 2 4"],
            ]
        );
    }
}
