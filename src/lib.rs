pub mod calibration;
mod image;

pub use image::*;

pub extern "C" fn register(collection: &mut minfac::ServiceCollection) {
    calibration::register_services(collection);
}
