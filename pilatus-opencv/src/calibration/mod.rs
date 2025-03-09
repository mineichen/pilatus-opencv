use std::{fmt::Debug, num::NonZeroU32, path::Path, sync::Arc};

use futures::stream::BoxStream;
use opencv::{
    calib3d::{self},
    core::{self, Mat, MatTrait, MatTraitConst, Point, Point2f, Point3_, Point3f, Size_, Vector},
    imgproc,
    prelude::CharucoBoardTraitConst,
};
use pilatus::device::{ActorError, ActorMessage};

mod intrinsic;
mod pixel_to_world;

pub use intrinsic::*;
pub use pixel_to_world::*;
use serde::{Deserialize, Serialize};
use tracing::trace;
use typed_floats::NonNaNFinite;

pub type CalibrationResult<T> = Result<T, CalibrationError>;
const THICKNESS: i32 = 2;

#[derive(Debug, Clone, thiserror::Error)]
pub enum CalibrationError {
    #[error("Not initialized")]
    NotInitialized,
    #[error("Not enough points found")]
    NotEnoughPoints { image: Mat, points: Vec<Point2f> },
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

pub struct CameraPosition {
    rvec: Mat,
    tvec: Mat,
}

impl Debug for CameraPosition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let r = [self.rvec.at::<f64>(0).expect("3x1 vec"), self.rvec.at::<f64>(1).expect("3x1 vec"), self.rvec.at::<f64>(2).expect("3x1 vec")];
        let t = [self.tvec.at::<f64>(0).expect("3x1 vec"), self.tvec.at::<f64>(1).expect("3x1 vec"), self.tvec.at::<f64>(2).expect("3x1 vec")];
        f.debug_struct("CameraPosition").field("rvec", &r).field("tvec", &t).finish()
    }
}

pub struct ExtrinsicCalibration {
    intrinsic: IntrinsicCalibration,
    position: CameraPosition,
    image_size: Size_<NonZeroU32>,
}



impl ExtrinsicCalibration {
    pub fn image_size(&self) -> Size_<NonZeroU32> {
        self.image_size
    }

    pub fn position(&self) -> &CameraPosition {
        &self.position
    }

    pub fn apply_offset(&mut self, offsets: &OrientationOffset) {
        for i in 0..3 {
            let rot = self.position.rvec.at_mut::<f64>(i as i32).unwrap_or_else(|_| panic!("created from calibration rot {i}"));
            *rot = *rot + offsets.angle[i].get();
        }

        for i in 0..3 {
            let trans = self.position.tvec.at_mut::<f64>(i as i32).unwrap_or_else(|_| panic!("created from calibration trans {i}"));
            *trans = *trans + offsets.translation[i].get();
        }       
    }

    pub fn build_pixel_to_world(&self) -> opencv::Result<PixelToWorldLut> {
        let transformer = self.pixel_to_world_transformer()?;
        let time = std::time::Instant::now();
        let lut = PixelToWorldLut::new(&transformer, self.image_size);
        trace!("Lut generation took {:?}", time.elapsed());
        Ok(lut)
    }

    pub fn pixel_to_world_transformer(&self) -> opencv::Result<PixelToWorldTransformer> {
        PixelToWorldTransformer::new(
            self.camera_matrix(),
            &self.position.rvec,
            &self.position.tvec,
            &self.dist_coeffs(),
        )
    }

    pub fn draw_debug_points(
        &self,
        image: &mut Mat,
        transformer: &PixelToWorldTransformer,
    ) -> opencv::Result<()> {
        for (_, pixel) in self.debug_points(transformer)? {
            draw_inforced_circle(image, &pixel)?;
        }
        let zero = self.project_points(&Vector::from_elem(Point3f::new(0., 0., 0.), 1))?;
        let zero = zero.at::<Point2f>(0)?;

        imgproc::circle(
            image,
            Point::new(zero.x.round() as _, zero.y.round() as _),
            14,
            core::Scalar::new(0.0, 0.0, 255.0, 1.0),
            THICKNESS,
            imgproc::LINE_8,
            0,
        )
    }

    fn debug_points<'a>(
        &'a self,
        transformer: &'a PixelToWorldTransformer,
    ) -> opencv::Result<Vec<(Point2f, Point2f)>> {
        let board = self.intrinsic.board()?;
        let square_len = board.get_square_length()?;
        let board = board.get_chessboard_size()?;
        ((-1)..(board.height + 2))
            .flat_map(|x| {
                (-1..(board.width + 2)).map(move |y| (x as f32 * square_len, y as f32 * square_len))
            })
            .map(|(x, y)| {
                let point_world = Point3f::new(x, y, 0.0);
                let points_world = Vector::from_slice(&[point_world]);
                let projected_points = self.project_points(&points_world)?;
                let coord = projected_points.at::<Point2f>(0)?;

               
                let new_accurate = transformer.transform_point(coord.x, coord.y); trace!("Pixel position for {points_world:?}: {coord:?}, Transformed back: {new_accurate:?})");

                Ok((Point2f::new(x, y), Point2f::new(coord.x, coord.y)))
            })
            .collect()
    }
    fn project_points(&self, points_world: &Vector<Point3f>) -> opencv::Result<Mat> {
        let mut projected_points = Mat::default();

        let points_world: Vector<Point3f> = points_world
            .iter()
            .map(|p| Point3_::new(p.y, p.x, p.z))
            .collect();

        calib3d::project_points_def(
            &points_world,
            &self.position.rvec,
            &self.position.tvec,
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

#[derive(Debug, Deserialize, Serialize, Clone, Default)]
#[serde(default, deny_unknown_fields)]
pub struct OrientationOffset {
    angle: [NonNaNFinite; 3],
    translation: [NonNaNFinite; 3],
}

fn draw_inforced_circle(image: &mut Mat, pixel: &Point2f) -> opencv::Result<()> {
    let pixel_quantisized = Point::new(pixel.x.round() as _, pixel.y.round() as _);
    imgproc::circle(
        image,
        pixel_quantisized,
        10,
        core::Scalar::new(255.0, 0.0, 0.0, 1.0),
        THICKNESS,
        imgproc::LINE_8,
        0,
    )?;
    imgproc::circle(
        image,
        pixel_quantisized,
        12,
        core::Scalar::new(0.0, 255.0, 0., 1.0),
        THICKNESS,
        imgproc::LINE_8,
        0,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handle_charuco() {
        match run_charuco() {
            Ok(x) => {}
            Err(CalibrationError::NotEnoughPoints { image, points }) => {
                let mut output = Vector::new();
                opencv::imgcodecs::imencode(".png", &image, &mut output, &Vector::new()).unwrap();
                std::fs::write("NotEnoughPoints.png", &output).unwrap();
                panic!("Not enouth points {points:?}");
            }
            i @ Err(_) => i.unwrap(),
        }
    }
    fn run_charuco() -> CalibrationResult<()> {
        let iter = ImageIterable::from_dir("../charuco_raw_images");
        //let iter = ImageIterable::from_dir("/Users/mineichen/Downloads/2Caps");
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
            println!("Before extrinsic");
            let extrinsic = intrinsic.clone().calibrate_extrinsic(&distorted)?;
            let world_transformer = extrinsic.pixel_to_world_transformer()?;
            for (label, mut image) in [("distorded", distorted), ("undistorted", undistorted)] {
                let path = target_dir.join(format!("{stem}_{label}.png"));

                // let horizontal_pixels = [[1894, 552], [1713, 797]];
                // for [x_pixel, y_pixel] in horizontal_pixels {
                //     let [x_world, y_world] =
                //         world_transformer.transform_point(x_pixel as f32, y_pixel as f32);
                //     println!("In line? {x_world}, {y_world}");
                //     draw_inforced_circle(
                //         &mut image,
                //         &Point2f::new(x_pixel as f32, y_pixel as f32),
                //     )?;
                // }

                extrinsic.draw_debug_points(&mut image, &world_transformer)?;
                let mut output = Vector::new();
                opencv::imgcodecs::imencode(".png", &image, &mut output, &Vector::new())?;
                std::fs::write(&path, &output).unwrap();
                //opencv::imgcodecs::imwrite(&path, &image, &Vector::new())?;
            }
        }
        Ok(())
    }
}
