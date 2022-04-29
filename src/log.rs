use env_logger::{Builder, Env};
use log::LevelFilter;
use std::io::Write;
use std::sync::Arc;
use std::time::Instant;

/// Builds the logger using the environment variable 'RUST_LOG' to determine the log level. Uses the
/// passed in `level` if the environment variable is not set.
pub fn build_pace_logger_for_level(level: LevelFilter) {
    let start_time = Arc::new(Instant::now());

    let env = Env::default().default_filter_or(level.as_str());
    let mut builder = Builder::from_env(env);
    builder
        .format(move |buf, record| {
            let elapsed = start_time.elapsed().as_millis();
            writeln!(
                buf,
                "c {:>6}.{:<03} [{}] - {}",
                elapsed / 1000,
                elapsed % 1000,
                record.level(),
                record.args()
            )
        })
        .init();
}

/// Builds the logger using the environment variable 'RUST_LOG' to determine the log level. If the
/// environment variable is not set, the passed in `default_level` is increased by `verbosity` many
/// levels and the result is used as the log level.
pub fn build_pace_logger_for_verbosity(default_level: LevelFilter, verbosity: usize) {
    let result_level = level_from_verbosity(default_level, verbosity);
    build_pace_logger_for_level(result_level);
}

fn level_from_verbosity(default_level: LevelFilter, verbosity: usize) -> LevelFilter {
    let default_level = usize_from_level(default_level);
    try_level_from_usize(default_level + verbosity).unwrap_or(LevelFilter::Trace)
}

fn usize_from_level(value: LevelFilter) -> usize {
    match value {
        LevelFilter::Off => 0,
        LevelFilter::Error => 1,
        LevelFilter::Warn => 2,
        LevelFilter::Info => 3,
        LevelFilter::Debug => 4,
        LevelFilter::Trace => 5,
    }
}

fn try_level_from_usize(value: usize) -> Option<LevelFilter> {
    match value {
        0 => Some(LevelFilter::Off),
        1 => Some(LevelFilter::Error),
        2 => Some(LevelFilter::Warn),
        3 => Some(LevelFilter::Info),
        4 => Some(LevelFilter::Debug),
        5 => Some(LevelFilter::Trace),
        _ => None,
    }
}

#[cfg(feature = "test-case")]
#[cfg(test)]
mod test_cases {
    use super::*;
    use log::LevelFilter;
    use test_case::test_case;

    #[test_case(LevelFilter::Off, 5 => LevelFilter::Trace)]
    #[test_case(LevelFilter::Warn, 1 => LevelFilter::Info)]
    #[test_case(LevelFilter::Error, 0 => LevelFilter::Error)]
    #[test_case(LevelFilter::Trace, 1 => LevelFilter::Trace)]
    fn test_level_from_verbosity(default: LevelFilter, verbosity: usize) -> LevelFilter {
        level_from_verbosity(default, verbosity)
    }
}
