use std::collections::HashMap;
use std::fmt::Display;

pub struct KeyedBuffer {
    //TODO: Use IndexMap for stable element order and remove columns vector
    entries: HashMap<String, String>,
    columns: Vec<String>, // separate vector is needed for stable order of columns when iterating
    is_empty: bool,
}

impl KeyedBuffer {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            columns: vec![],
            is_empty: true,
        }
    }

    /// Write one entry into the buffer
    pub fn write(&mut self, column: impl Display, value: impl Display) {
        let column_str = column.to_string();
        let old_value = self.entries.insert(column_str.clone(), value.to_string());
        self.is_empty = false;

        // new column
        if old_value.is_none() {
            self.columns.push(column_str);
        }
    }

    /// Returns all buffered values
    pub fn flush(&mut self) -> Vec<String> {
        // Replaces all values with empty strings instead of just removing them.
        // So that the column entries in the map are kept and we can check in O(1) if a column
        // is new inside of KeyedCsvWriter::write()
        self.is_empty = true;
        let mut csv_row = Vec::with_capacity(self.columns.len());
        for key in &self.columns {
            let old_value = self
                .entries
                .insert(key.clone(), String::new())
                .unwrap_or_default();
            csv_row.push(old_value);
        }

        csv_row
    }

    // Returns all columns
    pub fn get_columns(&self) -> &[String] {
        &self.columns
    }

    pub fn is_empty(&self) -> bool {
        self.is_empty
    }
}

impl Default for KeyedBuffer {
    fn default() -> Self {
        Self::new()
    }
}
