use std::ops::Deref;

use imbuf::{ImageChannel, PixelType};
use opencv::{boxed_ref::BoxedRef, core::Mat};
use pilatus_engineering::image::{DynamicImage, Image};

pub struct BorrowImage<'mat>(BoxedRef<'mat, Mat>);
impl<'a> Deref for BorrowImage<'a> {
    type Target = BoxedRef<'a, Mat>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'mat, 'img: 'mat> TryFrom<&'img DynamicImage> for BorrowImage<'mat> {
    type Error = opencv::Error;

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
            _ => Err(opencv::Error::new(
                opencv::core::Code::StsUnsupportedFormat.into(),
                "unsupported format",
            )),
        }
    }
}

impl<'mat, 'img: 'mat, T: PixelType + 'static + opencv::core::DataType>
    TryFrom<&'img ImageChannel<T>> for BorrowImage<'mat>
{
    type Error = opencv::Error;

    fn try_from(img: &'img ImageChannel<T>) -> Result<Self, Self::Error> {
        let (width, height) = img.dimensions();
        Ok(BorrowImage(Mat::new_rows_cols_with_data(
            height.get() as _,
            width.get() as _,
            img.buffer() as _,
        )?))
    }
}
impl<'mat, 'img: 'mat, T: PixelType + 'static + opencv::core::DataType> TryFrom<&'img Image<T, 1>>
    for BorrowImage<'mat>
{
    type Error = opencv::Error;

    fn try_from(img: &'img Image<T, 1>) -> Result<Self, Self::Error> {
        let (width, height) = img.dimensions();
        Ok(BorrowImage(Mat::new_rows_cols_with_data(
            height.get() as _,
            width.get() as _,
            img.buffer() as _,
        )?))
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use opencv::core::Vector;
    use opencv::prelude::MatTraitConst;
    use opencv::prelude::MatTraitConstManual;

    use super::*;

    fn imbuf_image_to_borrowed_mat_preserves_dims_and_content<T>(
        pixels: Vec<T>,
        decode_png_to_vec: impl Fn(&[u8]) -> Vec<T>,
    ) where
        T: PixelType + opencv::core::DataType + std::fmt::Debug + PartialEq + 'static,
    {
        let width = NonZeroU32::new(3).unwrap();
        let height = NonZeroU32::new(2).unwrap();
        let img: Image<T, 1> = Image::new_vec(pixels, width, height);

        let borrowed: BorrowImage<'_> = (&img).try_into().unwrap();
        let mat = &*borrowed;

        assert_eq!(img.width(), width);
        assert_eq!(img.height(), height);
        assert_eq!(mat.rows(), height.get() as i32);
        assert_eq!(mat.cols(), width.get() as i32);

        let mat_data = mat.data_typed::<T>().unwrap();
        assert_eq!(mat_data, img.buffer());
        assert_eq!(mat_data.as_ptr() as usize, img.buffer().as_ptr() as usize);

        // Encode the borrowed Mat as PNG, reload via the `image` crate, and compare decoded pixels
        // with the original image buffer.
        let mut png = Vector::<u8>::new();
        opencv::imgcodecs::imencode(".png", mat, &mut png, &Vector::new()).unwrap();
        let decoded = decode_png_to_vec(png.as_slice());
        assert_eq!(decoded, img.buffer());
    }

    #[test]
    fn imbuf_image_u8_to_borrowed_mat_preserves_dims_and_content() {
        imbuf_image_to_borrowed_mat_preserves_dims_and_content(vec![0u8, 1, 2, 3, 4, 5], |png| {
            match image::load_from_memory_with_format(png, image::ImageFormat::Png).unwrap() {
                image::DynamicImage::ImageLuma8(img) => img.into_vec(),
                other => panic!("Expected Luma8 PNG, got {other:?}"),
            }
        });
    }

    #[test]
    fn imbuf_image_u16_to_borrowed_mat_preserves_dims_and_content() {
        imbuf_image_to_borrowed_mat_preserves_dims_and_content(
            vec![10u16, 20, 30, 40, 50, 60],
            |png| match image::load_from_memory_with_format(png, image::ImageFormat::Png).unwrap() {
                image::DynamicImage::ImageLuma16(img) => img.into_vec(),
                other => panic!("Expected Luma16 PNG, got {other:?}"),
            },
        );
    }
}
