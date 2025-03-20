pub mod calibration;
mod image;
use std::backtrace::Backtrace;

pub use image::*;

#[derive(Debug, thiserror::Error)]
#[error("{source:?}: {trace}")]
pub struct Error {
    source: opencv::Error,
    trace: String,
}

impl From<opencv::Error> for Error {
    fn from(value: opencv::Error) -> Self {
        Self {
            source: value,
            trace: std::backtrace::Backtrace::capture().to_string(),
        }
    }
}
