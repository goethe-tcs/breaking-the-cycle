use super::bench_writer::BenchWriter;
use super::keyed_buffer::KeyedBuffer;
use itertools::Itertools;
use std::fmt::Display;
use std::fs::File;
use std::io::{self, Result, Stdout, Write};
use std::path::Path;

pub struct KeyedWriter<W: Write + 'static> {
    line_buffer: KeyedBuffer,
    inner_writer: W,
    line_terminator: char,
    entry_formatter: Box<dyn Fn(String, String) -> String>,
}

impl<W: Write + 'static> KeyedWriter<W> {
    pub fn new(inner_writer: W, line_terminator: char) -> Self {
        Self::new_with_formatter(inner_writer, line_terminator, Self::default_formatter)
    }

    pub fn new_with_formatter<F>(inner_writer: W, line_terminator: char, formatter: F) -> Self
    where
        F: Fn(String, String) -> String + 'static,
    {
        Self {
            line_buffer: KeyedBuffer::new(),
            inner_writer,
            line_terminator,
            entry_formatter: Box::new(formatter),
        }
    }

    pub fn default_formatter(column: String, value: String) -> String {
        format!(" {}: {:>12} |", column, value)
    }

    fn write(&mut self, column: impl Display, value: impl Display) {
        self.line_buffer.write(column, value);
    }

    fn end_line(&mut self) -> Result<()> {
        let mut line = self
            .line_buffer
            .flush()
            .into_iter()
            .zip(self.line_buffer.get_columns())
            .map(|(val, col)| (*self.entry_formatter)(col.clone(), val))
            .join("");

        line.push(self.line_terminator);
        self.inner_writer.write_all(line.as_bytes())?;
        self.inner_writer.flush()?;
        Ok(())
    }
}

impl KeyedWriter<Stdout> {
    pub fn new_std_out_writer() -> Self {
        let std_out_handler = io::stdout();

        Self::new(std_out_handler, '\n')
    }
}

impl KeyedWriter<File> {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::create(path)?;

        Ok(Self::new(file, '\n'))
    }
}

impl<W: Write + 'static> BenchWriter for KeyedWriter<W> {
    fn write(&mut self, column: impl Display, value: impl Display) -> Result<()> {
        self.write(column, value);
        Ok(())
    }

    fn end_design_point(&mut self) -> Result<()> {
        self.end_line()
    }

    fn end_graph_section(&mut self) -> Result<()> {
        Ok(())
    }
}

impl<W: Write + 'static> Drop for KeyedWriter<W> {
    /// end the current line in destructor
    fn drop(&mut self) {
        if !self.line_buffer.is_empty() {
            self.end_line().expect("Failed to flush line in destructor");
        }
    }
}

#[cfg(test)]
mod tests_keyed_writer {
    use super::*;
    use itertools::Itertools;
    use std::io::Read;

    #[cfg(feature = "tempfile")]
    #[test]
    fn test_inconsistent_writes() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.txt");

        // test destructor
        {
            let file_writer = File::create(file_path.clone()).unwrap();
            let mut writer = KeyedWriter::new(file_writer, '\n');

            writer.write("col_1", "val_1_1");
            writer.write("col_2", "val_1_2");
            writer.end_line().unwrap();

            writer.write("col_3", "val_2_3");
            writer.write("col_1", "val_2_1");
        }

        let mut file = File::open(file_path).unwrap();
        let mut text = String::new();
        file.read_to_string(&mut text).unwrap();
        let lines = text.split('\n').collect_vec();
        assert_eq!(
            lines,
            vec![
                " col_1:      val_1_1 | col_2:      val_1_2 |",
                " col_1:      val_2_1 | col_2:              | col_3:      val_2_3 |",
                ""
            ]
        );
    }

    #[cfg(feature = "tempfile")]
    #[test]
    fn test_custom_formater() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.txt");

        // test destructor
        {
            let entry_formatter = |col, val| format!("{} = {}|", col, val);
            let file_writer = File::create(file_path.clone()).unwrap();
            let mut writer = KeyedWriter::new_with_formatter(file_writer, ';', entry_formatter);

            writer.write("col_1", "val_1_1");
            writer.write("col_2", "val_1_2");
            writer.end_line().unwrap();

            writer.write("col_3", "val_2_3");
            writer.write("col_1", "val_2_1");
        }

        let mut file = File::open(file_path).unwrap();
        let mut text = String::new();
        file.read_to_string(&mut text).unwrap();
        assert_eq!(
            text,
            "col_1 = val_1_1|col_2 = val_1_2|;col_1 = val_2_1|col_2 = |col_3 = val_2_3|;",
        );
    }
}
