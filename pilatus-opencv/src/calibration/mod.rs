use std::{path::Path, sync::Arc};

use futures::stream::BoxStream;
use opencv::{
    calib3d::{self},
    core::{self, Mat, MatTraitConst, Point, Point2f, Point3f, Size_, Vector},
    imgproc,
};
use pilatus::device::{ActorError, ActorMessage};

mod intrinsic;
mod pixel_to_world;

pub use intrinsic::*;
pub use pixel_to_world::*;
use tracing::trace;

pub type CalibrationResult<T> = Result<T, CalibrationError>;

#[derive(Debug, Clone, thiserror::Error)]
pub enum CalibrationError {
    #[error("Not initialized")]
    NotInitialized,
    #[error("{0:?}")]
    Other(Arc<opencv::Error>),
}

impl From<opencv::Error> for CalibrationError {
    fn from(value: opencv::Error) -> Self {
        CalibrationError::Other(Arc::new(value))
    }
}

pub struct StreamProjectorMessage;

impl ActorMessage for StreamProjectorMessage {
    type Output = BoxStream<'static, CalibrationResult<PixelToWorldLut>>;
    type Error = std::convert::Infallible;
}

#[non_exhaustive]
#[derive(Default)]
pub struct CalibrationDetailMessage {}

#[derive(Debug, Clone, Copy, thiserror::Error)]
#[error("There was not successful calibration to be evaluated")]
pub struct NoCalibrationDetailsError;

impl From<NoCalibrationDetailsError> for ActorError<NoCalibrationDetailsError> {
    fn from(value: NoCalibrationDetailsError) -> Self {
        ActorError::Custom(value)
    }
}

impl ActorMessage for CalibrationDetailMessage {
    type Output = Vec<u8>;
    type Error = NoCalibrationDetailsError;
}

pub struct ExtrinsicCalibration {
    intrinsic: IntrinsicCalibration,
    rvec: Mat,
    tvec: Mat,
    image_size: Size_<i32>,
}

pub struct ImageIterable {
    paths: Vec<String>,
}

impl ImageIterable {
    pub fn from_dir(path: impl AsRef<Path>) -> Self {
        let mut paths: Vec<_> = std::fs::read_dir(path)
            .into_iter()
            .flatten()
            .filter_map(|entry| {
                let entry = entry.ok()?;
                let path = entry.path();
                let ext = path.extension()?;
                if ext == "jpg" || ext == "jpeg" || ext == "png" || ext == "tiff" {
                    Some(path.to_str()?.to_owned())
                } else {
                    None
                }
            })
            .collect();
        paths.sort_unstable();
        Self { paths }
    }

    fn iter_images(&self) -> impl Iterator<Item = (&'_ str, opencv::Result<Mat>)> {
        self.paths.iter().map(|p| {
            (
                p.as_str(),
                opencv::imgcodecs::imread(p, opencv::imgcodecs::IMREAD_COLOR),
            )
        })
    }
}

pub struct Undistorter {
    map1: Mat,
    map2: Mat,
}

impl Undistorter {
    fn undistort(&self, distorted: &Mat) -> opencv::Result<Mat> {
        let mut undistorted = Mat::default();
        // Remap the image
        imgproc::remap(
            distorted,
            &mut undistorted,
            &self.map1,
            &self.map2,
            imgproc::INTER_LINEAR,
            core::BORDER_CONSTANT,
            core::Scalar::all(0.0),
        )?;
        Ok(undistorted)
    }
}

impl ExtrinsicCalibration {
    pub fn build_world_to_pixel(&self) -> opencv::Result<PixelToWorldLut> {
        let transformer = self.pixel_to_world_transformer()?;
        let time = std::time::Instant::now();
        let lut = PixelToWorldLut::new(
            &transformer,
            (self.image_size.width as u32)
                .try_into()
                .expect("Images are always wider 0"),
            (self.image_size.height as u32)
                .try_into()
                .expect("Images are always higher 0"),
        );
        trace!("Lut generation took {:?}", time.elapsed());
        Ok(lut)
    }

    pub fn pixel_to_world_transformer(&self) -> opencv::Result<PixelToWorldTransformer> {
        PixelToWorldTransformer::new(
            self.camera_matrix(),
            &self.rvec,
            &self.tvec,
            &self.dist_coeffs(),
        )
    }

    pub fn draw_debug_points(
        &self,
        image: &mut Mat,
        transformer: &PixelToWorldLut,
    ) -> opencv::Result<()> {
        let thickness = 2;
        for (_, pixel) in self.debug_points(transformer)? {
            let pixel_quantisized = Point::new(pixel.x.round() as _, pixel.y.round() as _);
            imgproc::circle(
                image,
                pixel_quantisized,
                10,
                core::Scalar::new(255.0, 0.0, 0.0, 1.0),
                thickness,
                imgproc::LINE_8,
                0,
            )?;
            imgproc::circle(
                image,
                pixel_quantisized,
                12,
                core::Scalar::new(0.0, 255.0, 0., 1.0),
                thickness,
                imgproc::LINE_8,
                0,
            )?;
        }
        let zero = self.project_points(&Vector::from_elem(Point3f::new(0., 0., 0.), 1))?;
        let zero = zero.at::<Point2f>(0)?;

        imgproc::circle(
            image,
            Point::new(zero.x.round() as _, zero.y.round() as _),
            14,
            core::Scalar::new(0.0, 0.0, 255.0, 1.0),
            thickness,
            imgproc::LINE_8,
            0,
        )
    }

    fn debug_points<'a>(
        &'a self,
        lut: &'a PixelToWorldLut,
    ) -> opencv::Result<Vec<(Point2f, Point2f)>> {
        let transformer = &self.pixel_to_world_transformer()?;
        let square_len = self.intrinsic.board_square_length()?;
        (-1..7)
            .flat_map(|x| (-1..9).map(move |y| (x as f32 * square_len, y as f32 * square_len)))
            .map(|(x, y)| {
                let point_world = Point3f::new(x, y, 0.0);
                let points_world = Vector::from_slice(&[point_world]);
                let projected_points = self.project_points(&points_world)?;
                let coord = projected_points.at::<Point2f>(0)?;

                trace!("Pixel position for {points_world:?}: {:?}", coord);
                let pixel_x_rounded = coord.x.round();
                let pixel_y_rounded = coord.y.round();
                let new_accurate = transformer.transform_point(coord.x, coord.y);
                if (0..self.image_size.width).contains(&(pixel_x_rounded as i32))
                    && (0..self.image_size.height).contains(&(pixel_y_rounded as i32))
                {
                    let new_lut = lut.get(pixel_x_rounded as _, pixel_y_rounded as _);
                    trace!("World coordinates: (accurate: {new_accurate:?}, lut: {new_lut:?})");
                } else {
                    trace!("Not within the image (accurate: {new_accurate:?})");
                }

                Ok((Point2f::new(x, y), Point2f::new(coord.x, coord.y)))
            })
            .collect()
    }
    fn project_points(&self, points_world: &Vector<Point3f>) -> opencv::Result<Mat> {
        let mut projected_points = Mat::default();

        calib3d::project_points_def(
            &points_world,
            &self.rvec,
            &self.tvec,
            &self.camera_matrix(),
            &self.dist_coeffs(),
            &mut projected_points,
        )?;

        Ok(projected_points)
    }

    pub fn camera_matrix(&self) -> &Mat {
        self.intrinsic.camera_matrix()
    }
    pub fn dist_coeffs(&self) -> &Mat {
        self.intrinsic.dist_coeffs()
    }
}

// In your main function, after getting rvec and tvec from solvePnP:

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_charuco() -> opencv::Result<()> {
        let iter = ImageIterable::from_dir("../charuco_raw_images");
        let intrinsic =
            IntrinsicCalibration::create(iter.iter_images().map(|(_, i)| i), Default::default())?;
        let target_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("target");
        for (path, distorted) in iter.iter_images() {
            let stem = Path::new(path)
                .file_stem()
                .and_then(|s| s.to_str())
                .expect("Works as it is created from string above");
            let distorted = distorted?;
            let undistorter = intrinsic.create_undistorter(distorted.size()?)?;
            let undistorted = undistorter.undistort(&distorted)?;
            let extrinsic = intrinsic.clone().calibrate_extrinsic(&distorted)?;
            let world_transformer = extrinsic.build_world_to_pixel()?;
            for (label, mut image) in [("distorded", distorted), ("undistorted", undistorted)] {
                let path = target_dir.join(format!("{stem}_{label}.png"));

                extrinsic.draw_debug_points(&mut image, &world_transformer)?;
                let mut output = Vector::new();
                opencv::imgcodecs::imencode(".png", &image, &mut output, &Vector::new())?;
                std::fs::write(path, &output).unwrap();
                //opencv::imgcodecs::imwrite(&path, &image, &Vector::new())?;
            }
        }
        Ok(())
    }
}
