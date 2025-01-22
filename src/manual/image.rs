use std::ops::Deref;

use opencv::{boxed_ref::BoxedRef, core::Mat};
use pilatus_engineering::image::{DynamicImage, GenericImage};

pub struct BorrowImage<'mat>(BoxedRef<'mat, Mat>);
impl<'a> Deref for BorrowImage<'a> {
    type Target = BoxedRef<'a, Mat>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(thiserror::Error, Debug)]
#[error("Conversion Error {0:?}")]
pub struct ConvertImageError(#[from] Option<opencv::Error>);

impl<'mat, 'img: 'mat> TryFrom<&'img DynamicImage> for BorrowImage<'mat> {
    type Error = ConvertImageError;

    fn try_from(value: &'img DynamicImage) -> Result<Self, Self::Error> {
        match value {
            DynamicImage::Luma8(img) => img.try_into(),
            DynamicImage::Luma16(img) => img.try_into(),
            _ => Err(ConvertImageError(None)),
        }
    }
}

impl<'mat, 'img: 'mat, T: Clone + 'static + opencv::core::DataType>
    TryFrom<&'img GenericImage<T, 1>> for BorrowImage<'mat>
{
    type Error = ConvertImageError;

    fn try_from(img: &'img GenericImage<T, 1>) -> Result<Self, Self::Error> {
        let (width, height) = img.dimensions();
        Ok(BorrowImage(Mat::new_rows_cols_with_data(
            width.get() as _,
            height.get() as _,
            img.buffer() as _,
        )?))
    }
}
