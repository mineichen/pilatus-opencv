use opencv::core::Mat;

pub trait IntoImbuf {
    fn try_into_imbuf(self) -> opencv::Result<imbuf::DynamicImage>;
}

impl IntoImbuf for Mat {
    fn try_into_imbuf(self) -> opencv::Result<imbuf::DynamicImage> {
        Err(opencv::Error::new(1, "NotImplementd"))
    }
}
