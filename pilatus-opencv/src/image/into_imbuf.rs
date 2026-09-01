use std::{
    mem::transmute,
    num::{NonZeroU32, NonZeroU8},
    slice,
};

use imbuf::{
    DynamicImage, DynamicSize, ImageChannel, ImageChannelVTable, PixelTypePrimitive,
    UnsafeImageChannel,
};
use opencv::{
    boxed_ref::BoxedRef,
    core::{Code, Mat, MatTraitConst, Range, Vector, CV_16U, CV_32F, CV_8U, CV_MAT_DEPTH_MASK},
    mod_prelude::OpenCVIntoExternContainer,
    prelude::MatTraitConstManual,
};

pub trait TryIntoImbuf {
    fn try_into_imbuf(self) -> opencv::Result<imbuf::DynamicImage>;
}

impl<H: MatHeader> TryIntoImbuf for H {
    fn try_into_imbuf(self) -> opencv::Result<DynamicImage> {
        match self.typ() & CV_MAT_DEPTH_MASK {
            CV_8U => dynamic_image::<H, u8>(self),
            CV_16U => dynamic_image::<H, u16>(self),
            CV_32F => dynamic_image::<H, f32>(self),
            _ => Err(opencv::Error::new(
                Code::StsUnsupportedFormat.into(),
                "Mat depth must be CV_8U, CV_16U or CV_32F",
            )),
        }
    }
}

trait MatHeader: MatTraitConst + 'static + Sized {
    fn from_owned(mat: Mat) -> Self;

    fn shallow_copy(&self) -> opencv::Result<Mat> {
        Mat::copy(self).and_then(OpenCVIntoExternContainer::opencv_into_extern_container)
    }

    fn shared(&self) -> opencv::Result<Self> {
        Ok(Self::from_owned(self.shallow_copy()?))
    }

    fn unique(&self) -> opencv::Result<Self> {
        Ok(Self::from_owned(self.try_clone()?))
    }
}

impl MatHeader for Mat {
    fn from_owned(mat: Mat) -> Self {
        mat
    }
}

impl MatHeader for BoxedRef<'static, Mat> {
    fn from_owned(mat: Mat) -> Self {
        BoxedRef::from(mat)
    }
}

fn dynamic_image<H: MatHeader, P: PixelTypePrimitive>(mat: H) -> opencv::Result<DynamicImage> {
    let dims = mat.dims();
    if !(2..=3).contains(&dims) || mat.data().is_null() {
        return Err(opencv::Error::new(
            Code::StsBadArg.into(),
            "Mat must be a non-empty 2d or 3d matrix",
        ));
    }
    if !mat.is_physically_contiguous()? {
        return Err(opencv::Error::new(
            Code::StsUnmatchedSizes.into(),
            "Mat is not continuous",
        ));
    }

    if dims == 2 {
        let width = nonzero_dimension(mat.cols())?;
        let height = nonzero_dimension(mat.rows())?;
        let pixel_elements = nonzero_pixel_elements(mat.channels())?;
        let channel = unsafe { into_channel::<H, P>(Box::new(mat), width, height, pixel_elements) };
        Ok(DynamicImage::from_channels(channel.into(), []))
    } else {
        if mat.channels() != 1 {
            return Err(opencv::Error::new(
                Code::StsUnsupportedFormat.into(),
                "A planar Mat must have exactly 1 channel",
            ));
        }
        let sizes = &*mat.mat_size();
        let planes = nonzero_pixel_elements(sizes[0])?;
        let width = nonzero_dimension(sizes[2])?;
        let height = nonzero_dimension(sizes[1])?;
        let channels = (0..planes.get())
            .map(|plane| planar_channel::<H, P>(&mat, plane as i32, width, height))
            .collect::<opencv::Result<Vec<_>>>()?;
        let mut channels = channels.into_iter();
        let Some(first) = channels.next() else {
            return Err(opencv::Error::new(
                Code::StsBadArg.into(),
                "Mat must have at least one plane",
            ));
        };
        Ok(DynamicImage::from_channels(
            first.into(),
            channels.map(Into::into),
        ))
    }
}

trait ChannelStorage<P: PixelTypePrimitive>: MatHeader {
    const VTABLE: &'static ImageChannelVTable<P>;
}

impl<H: MatHeader, P: PixelTypePrimitive> ChannelStorage<P> for H {
    const VTABLE: &'static ImageChannelVTable<P> = &ImageChannelVTable {
        clone: clone_mat_channel::<H, P>,
        make_mut: make_mat_channel_mut::<H, P>,
        drop: drop_mat_channel::<H, P>,
    };
}

unsafe fn into_channel<H: MatHeader, P: PixelTypePrimitive>(
    mat: Box<H>,
    width: NonZeroU32,
    height: NonZeroU32,
    pixel_elements: NonZeroU8,
) -> ImageChannel<DynamicSize<P>> {
    let ptr = mat.data() as *const P;
    let mat = Box::into_raw(mat);
    let channel = unsafe {
        UnsafeImageChannel::new_with_vtable(
            ptr,
            width,
            height,
            pixel_elements,
            <H as ChannelStorage<P>>::VTABLE,
            mat as *mut (),
        )
    };
    unsafe { transmute(channel) }
}

fn planar_channel<H: MatHeader, P: PixelTypePrimitive>(
    mat: &H,
    plane: i32,
    width: NonZeroU32,
    height: NonZeroU32,
) -> opencv::Result<ImageChannel<DynamicSize<P>>> {
    let mut ranges = Vector::<Range>::new();
    ranges.push(Range::new(plane, plane + 1)?);
    ranges.push(Range::all()?);
    ranges.push(Range::all()?);
    let view = Mat::ranges(mat, &ranges)?.opencv_into_extern_container()?;
    Ok(unsafe { into_channel(Box::new(H::from_owned(view)), width, height, NonZeroU8::MIN) })
}

unsafe extern "C" fn clone_mat_channel<H: MatHeader, P: PixelTypePrimitive>(
    image: &UnsafeImageChannel<P>,
) -> UnsafeImageChannel<P> {
    let mat = unsafe { &*(image.data as *const H) };
    match mat.shared() {
        Ok(shared) => unsafe {
            UnsafeImageChannel::new_with_vtable(
                image.ptr,
                image.width,
                image.height,
                image.pixel_elements,
                image.vtable,
                Box::into_raw(Box::new(shared)) as *mut (),
            )
        },
        Err(error) => {
            tracing::warn!(%error, "Cloning channel as Vec copy, cannot share Mat buffer");
            copied_channel(image)
        }
    }
}

unsafe extern "C" fn make_mat_channel_mut<H: MatHeader, P: PixelTypePrimitive>(
    image: &mut UnsafeImageChannel<P>,
) {
    let mat = unsafe { &*(image.data as *const H) };
    *image = match mat.unique() {
        Ok(owned) => {
            let ptr = owned.data() as *const P;
            let data = Box::into_raw(Box::new(owned)) as *mut ();
            unsafe {
                UnsafeImageChannel::new_with_vtable(
                    ptr,
                    image.width,
                    image.height,
                    image.pixel_elements,
                    image.vtable,
                    data,
                )
            }
        }
        Err(error) => {
            tracing::warn!(%error, "Cloning channel as Vec copy, cannot clone Mat buffer");
            copied_channel(image)
        }
    };
}

unsafe extern "C" fn drop_mat_channel<H, P>(image: &mut UnsafeImageChannel<P>) {
    drop(unsafe { Box::from_raw(image.data as *mut H) });
}

fn copied_channel<P: PixelTypePrimitive>(image: &UnsafeImageChannel<P>) -> UnsafeImageChannel<P> {
    let len = image.width.get() as usize
        * image.height.get() as usize
        * image.pixel_elements.get() as usize;
    let data = unsafe { slice::from_raw_parts(image.ptr, len) }.to_vec();
    UnsafeImageChannel::new_vec(data, image.width, image.height, image.pixel_elements)
}

fn nonzero_dimension(value: i32) -> opencv::Result<NonZeroU32> {
    u32::try_from(value)
        .ok()
        .and_then(NonZeroU32::new)
        .ok_or_else(|| {
            opencv::Error::new(Code::StsOutOfRange.into(), "Mat dimension must be positive")
        })
}

fn nonzero_pixel_elements(value: i32) -> opencv::Result<NonZeroU8> {
    u8::try_from(value)
        .ok()
        .and_then(NonZeroU8::new)
        .ok_or_else(|| {
            opencv::Error::new(
                Code::StsOutOfRange.into(),
                "Mat channels must be in 1..=255",
            )
        })
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use imbuf::{DynamicImageChannel, Image};
    use opencv::core::{Code, Mat, VecN};
    use opencv::mod_prelude::OpenCVIntoExternContainer;
    use testresult::TestResult;

    use super::*;
    use crate::BorrowImage;

    fn dimensions(width: u32, height: u32) -> (NonZeroU32, NonZeroU32) {
        (
            NonZeroU32::new(width).unwrap(),
            NonZeroU32::new(height).unwrap(),
        )
    }

    fn boxed(mat: Mat) -> BoxedRef<'static, Mat> {
        BoxedRef::from(mat)
    }

    #[test]
    fn gray_u8_shares_buffer() {
        let mat = boxed(Mat::from_slice_2d(&[[1u8, 2, 3], [4, 5, 6]]).unwrap());
        let data_ptr = mat.data() as usize;
        let dynamic = mat.try_into_imbuf().unwrap();

        assert_eq!(dynamic.len_nonzero().get(), 1);
        let DynamicImageChannel::U8(channel) = dynamic.first() else {
            panic!("Expected U8 channel");
        };
        assert_eq!(channel.pixel_elements().get(), 1);
        assert_eq!(channel.dimensions(), dimensions(3, 2));
        assert_eq!(channel.buffer_flat(), &[1, 2, 3, 4, 5, 6]);
        assert_eq!(channel.buffer_flat().as_ptr() as usize, data_ptr);
    }

    #[test]
    fn gray_u16_shares_buffer() {
        let mat = Mat::from_slice_2d(&[[10u16, 20], [30, 40]]).unwrap();
        let data_ptr = mat.data() as usize;
        let dynamic = mat.try_into_imbuf().unwrap();

        assert_eq!(dynamic.len_nonzero().get(), 1);
        let DynamicImageChannel::U16(channel) = dynamic.first() else {
            panic!("Expected U16 channel");
        };
        assert_eq!(channel.pixel_elements().get(), 1);
        assert_eq!(channel.dimensions(), dimensions(2, 2));
        assert_eq!(channel.buffer_flat(), &[10, 20, 30, 40]);
        assert_eq!(channel.buffer_flat().as_ptr() as usize, data_ptr);
    }

    #[test]
    fn f32_shares_buffer() {
        let mat = Mat::from_slice_2d(&[[0.5f32, 1.5], [2.5, 3.5]]).unwrap();
        let data_ptr = mat.data() as usize;
        let dynamic = mat.try_into_imbuf().unwrap();

        let DynamicImageChannel::F32(channel) = dynamic.first() else {
            panic!("Expected F32 channel");
        };
        assert_eq!(channel.pixel_elements().get(), 1);
        assert_eq!(channel.dimensions(), dimensions(2, 2));
        assert_eq!(channel.buffer_flat(), &[0.5, 1.5, 2.5, 3.5]);
        assert_eq!(channel.buffer_flat().as_ptr() as usize, data_ptr);
    }

    #[test]
    fn interleaved_channels_are_a_single_dynamic_channel() -> TestResult {
        let mat = Mat::from_slice_2d(&[[VecN([1u8, 2, 3]), VecN([4, 5, 6])]]).unwrap();
        let dynamic = mat.try_into_imbuf().unwrap();

        assert_eq!(dynamic.len_nonzero().get(), 1);
        assert!(Image::<u8, 3>::try_from(dynamic.clone()).is_err());
        let interleaved = Image::<[u8; 3], 1>::try_from(dynamic.clone())?;
        let [channel] = interleaved.into_channels();
        assert_eq!(channel.pixel_elements().get(), 3);
        assert_eq!(channel.dimensions(), dimensions(2, 1));
        assert_eq!(channel.buffer_flat(), &[1, 2, 3, 4, 5, 6]);
        Ok(())
    }

    #[test]
    fn planar_mat_maps_to_planar_image() {
        let flat =
            Mat::from_slice_2d(&[[1u16, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]).unwrap();
        let mat: Mat = flat
            .reshape_nd(1, &[3, 2, 2])
            .unwrap()
            .opencv_into_extern_container_nofail();
        assert_eq!(mat.dims(), 3);

        let data_ptr = mat.data() as usize;
        let dynamic = mat.try_into_imbuf().unwrap();
        assert!(dynamic.is_uniform());
        assert_eq!(dynamic.len_nonzero().get(), 3);

        let DynamicImageChannel::U16(first) = dynamic.first() else {
            panic!("Expected U16 channel");
        };
        assert_eq!(first.pixel_elements().get(), 1);
        assert_eq!(first.buffer_flat().as_ptr() as usize, data_ptr);

        let image = Image::<u16, 3>::try_from(dynamic).unwrap();
        assert_eq!(image.dimensions(), dimensions(2, 2));
        assert_eq!(
            image.buffers(),
            [
                &[1u16, 2, 3, 4][..],
                &[5, 6, 7, 8][..],
                &[9, 10, 11, 12][..]
            ]
        );
    }

    #[test]
    fn clone_shares_buffer_and_survives_drop_of_original() {
        let mat = boxed(Mat::from_slice_2d(&[[1u8, 2], [3, 4]]).unwrap());
        let data_ptr = mat.data() as usize;
        let dynamic = mat.try_into_imbuf().unwrap();
        let cloned = dynamic.clone();
        drop(dynamic);

        let DynamicImageChannel::U8(channel) = cloned.first() else {
            panic!("Expected U8 channel");
        };
        assert_eq!(channel.buffer_flat(), &[1, 2, 3, 4]);
        assert_eq!(channel.buffer_flat().as_ptr() as usize, data_ptr);
    }

    #[test]
    fn make_mut_copies_shared_data() {
        let mat = boxed(Mat::from_slice_2d(&[[1u8, 2], [3, 4]]).unwrap());
        let data_ptr = mat.data() as usize;
        let mut dynamic = mat.try_into_imbuf().unwrap();
        let cloned = dynamic.clone();

        let DynamicImageChannel::U8(channel) = &mut dynamic[0] else {
            panic!("Expected U8 channel");
        };
        channel.primitive_make_mut()[0] = 42;

        let DynamicImageChannel::U8(channel) = &dynamic[0] else {
            panic!("Expected U8 channel");
        };
        assert_eq!(channel.buffer_flat(), &[42, 2, 3, 4]);
        assert_ne!(channel.buffer_flat().as_ptr() as usize, data_ptr);

        let DynamicImageChannel::U8(channel) = &cloned[0] else {
            panic!("Expected U8 channel");
        };
        assert_eq!(channel.buffer_flat(), &[1, 2, 3, 4]);
        assert_eq!(channel.buffer_flat().as_ptr() as usize, data_ptr);
    }

    #[test]
    fn round_trip_through_borrowed_mat() {
        let mat = Mat::from_slice_2d(&[[1u16, 2], [3, 4]]).unwrap();
        let dynamic = mat.try_into_imbuf().unwrap();

        let borrowed: BorrowImage<'_> = (&dynamic).try_into().unwrap();
        let mat = &*borrowed;
        assert_eq!(mat.rows(), 2);
        assert_eq!(mat.cols(), 2);
        assert_eq!(mat.data_typed::<u16>().unwrap(), &[1, 2, 3, 4]);
    }

    #[test]
    fn empty_mat_is_rejected() {
        let error = boxed(Mat::default()).try_into_imbuf().unwrap_err();
        assert_eq!(error.code_as_enum(), Some(Code::StsBadArg));
    }

    #[test]
    fn unsupported_depth_is_rejected() {
        let mat = boxed(Mat::from_slice_2d(&[[1i32, 2], [3, 4]]).unwrap());
        let error = mat.try_into_imbuf().unwrap_err();
        assert_eq!(error.code_as_enum(), Some(Code::StsUnsupportedFormat));
    }

    #[test]
    fn non_continuous_mat_is_rejected() {
        let mat = Mat::from_slice_2d(&[[1u8, 2, 3], [4, 5, 6]]).unwrap();
        let column = boxed(mat.col(0).unwrap().opencv_into_extern_container_nofail());
        assert!(!column.is_continuous());

        let error = column.try_into_imbuf().unwrap_err();
        assert_eq!(error.code_as_enum(), Some(Code::StsUnmatchedSizes));
    }
}
