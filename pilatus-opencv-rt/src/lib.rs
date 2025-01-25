mod calibration;

pub extern "C" fn register(collection: &mut minfac::ServiceCollection) {
    calibration::register_services(collection);
}
