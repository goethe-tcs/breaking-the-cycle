use std::fmt::Display;

/// Used to buffer metrics of a design point
pub struct DesignPointBuffer {
    buffer: Vec<(String, String)>,
}

impl DesignPointBuffer {
    pub fn new() -> Self {
        Self { buffer: vec![] }
    }

    /// Write one entry into the buffer
    pub fn write(&mut self, column: impl Display, value: impl Display) {
        self.buffer.push((column.to_string(), value.to_string()));
    }

    /// Consume the buffer and return its contents
    pub fn flush(self) -> Vec<(String, String)> {
        self.buffer
    }
}

impl Default for DesignPointBuffer {
    fn default() -> Self {
        Self::new()
    }
}
