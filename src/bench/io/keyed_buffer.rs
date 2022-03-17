use std::collections::HashMap;
use std::fmt::Display;

/// Used to buffer metrics of a design point
pub struct KeyedBuffer {
    entries: HashMap<String, String>,
    keys: Vec<String>, // separate vector is needed for stable order of keys when iterating
    is_empty: bool,
}

impl KeyedBuffer {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            keys: vec![],
            is_empty: true,
        }
    }

    /// Write one entry into the buffer
    pub fn write(&mut self, key: impl Display, value: impl Display) {
        let key_str = key.to_string();
        let old_value = self.entries.insert(key_str.clone(), value.to_string());
        self.is_empty = false;

        // new key
        if old_value.is_none() {
            self.keys.push(key_str);
        }
    }

    /// Returns all buffered values
    pub fn flush(&mut self) -> Vec<String> {
        // Replaces all values with empty strings instead of just removing them.
        // So that the key entries in the map are kept and we can check in O(1) if a key
        // is new inside of KeyedCsvWriter::write()
        self.is_empty = true;
        let mut csv_row = Vec::with_capacity(self.keys.len());
        for key in &self.keys {
            let old_value = self
                .entries
                .insert(key.clone(), String::new())
                .unwrap_or_default();
            csv_row.push(old_value);
        }

        csv_row
    }

    /// Returns all keys
    pub fn get_keys(&self) -> &[String] {
        &self.keys
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
