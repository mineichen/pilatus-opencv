use std::ops::Deref;

use imbuf::{DynamicImageChannel, ImageChannel, LumaImage, PixelType};
use opencv::{boxed_ref::BoxedRef, core::Mat};
use pilatus_engineering::image::{DynamicImage, Image};

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
        match (
            value.len(),
            value.first().pixel_elements().get(),
            value.first(),
        ) {
            (1, 1, imbuf::DynamicImageChannel::U8(c)) => {
                c.try_cast::<u8>().expect("Checked in match").try_into()
            }
            (1, 1, imbuf::DynamicImageChannel::U16(c)) => {
                c.try_cast::<u16>().expect("Checked in match").try_into()
            }
            _ => Err(ConvertImageError(None)),
        }
    }
}

impl<'mat, 'img: 'mat, T: PixelType + 'static + opencv::core::DataType>
    TryFrom<&'img ImageChannel<T>> for BorrowImage<'mat>
{
    type Error = ConvertImageError;

    fn try_from(img: &'img ImageChannel<T>) -> Result<Self, Self::Error> {
        let (width, height) = img.dimensions();
        Ok(BorrowImage(Mat::new_rows_cols_with_data(
            width.get() as _,
            height.get() as _,
            img.buffer() as _,
        )?))
    }
}
impl<'mat, 'img: 'mat, T: PixelType + 'static + opencv::core::DataType> TryFrom<&'img Image<T, 1>>
    for BorrowImage<'mat>
{
    type Error = ConvertImageError;

    fn try_from(img: &'img Image<T, 1>) -> Result<Self, Self::Error> {
        let (width, height) = img.dimensions();
        Ok(BorrowImage(Mat::new_rows_cols_with_data(
            width.get() as _,
            height.get() as _,
            img.buffer() as _,
        )?))
    }
}
