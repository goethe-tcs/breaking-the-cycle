#[cfg(feature = "bb-stats")]
use crate::bench::io::keyed_buffer::KeyedBuffer;

#[derive(Clone, Debug)]
#[cfg_attr(not(feature = "bb-stats"), derive(Default))]
pub struct BBStats {
    #[cfg(feature = "bb-stats")]
    entered_at: [usize; 128],
}

#[cfg(not(feature = "bb-stats"))]
impl BBStats {
    pub fn new() -> Self {
        Self {}
    }

    pub fn entered_at(&mut self, _n: usize) {}
}

#[cfg(feature = "bb-stats")]
impl Default for BBStats {
    fn default() -> Self {
        Self {
            entered_at: [0usize; 128],
        }
    }
}

#[cfg(feature = "bb-stats")]
impl BBStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn entered_at(&mut self, n: usize) {
        self.entered_at[n] += 1;
    }

    pub fn write_to_buffer(&self, buffer: &mut KeyedBuffer) {
        let mut prev_size_group = 0;
        let mut size_group = 8;

        let total: usize = self.entered_at.iter().sum();

        while size_group < self.entered_at.len() {
            let n_calls: usize = self.entered_at[prev_size_group..=size_group].iter().sum();
            buffer.write(
                format!("graph_size_{}_entered", size_group),
                n_calls as f64 / total as f64,
            );

            prev_size_group = size_group;
            size_group *= 2;
        }
    }
}
