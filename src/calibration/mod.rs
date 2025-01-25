use std::{path::Path, sync::Arc};

use opencv::{
    calib3d::{self},
    core::{self, Mat, Point2f, Point3f, Size_, Vector},
    imgproc,
    objdetect::{
        get_predefined_dictionary, CharucoBoard, CharucoDetector, PredefinedDictionaryType,
    },
    prelude::{CharucoDetectorTraitConst, *},
};

mod pixel_to_world;

pub use pixel_to_world::*;

type VectorOfMat = Vector<Mat>;

pub enum CalibrationError {}

#[derive(Clone)]
pub struct IntrinsicCalibration {
    detector: Arc<CharucoDetector>,
    camera_matrix: Mat,
    dist_coeffs: Mat,
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
    pub fn create_image_iterator(path: impl AsRef<Path>) -> Self {
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

impl IntrinsicCalibration {
    pub fn create_undistorter(&self, image_size: core::Size_<i32>) -> opencv::Result<Undistorter> {
        let mut map1 = Mat::default();
        let mut map2 = Mat::default();

        // Initialize undistortion maps
        calib3d::init_undistort_rectify_map(
            &self.camera_matrix,
            &self.dist_coeffs,
            &Mat::default(),
            &self.camera_matrix, // Use same camera matrix for simplicity
            image_size,
            core::CV_32FC1,
            &mut map1,
            &mut map2,
        )?;
        Ok(Undistorter { map1, map2 })
    }

    pub fn create(
        images: impl IntoIterator<Item = opencv::Result<Mat>>,
    ) -> Result<IntrinsicCalibration, opencv::Error> {
        // Directory containing calibration images

        let square_size = 29.0; // mm
        let marker_size = 14.0; // mm

        // Create ChArUco board

        let dictionary = get_predefined_dictionary(PredefinedDictionaryType::DICT_5X5_1000)?;
        let board = CharucoBoard::new_def(
            opencv::core::Size::new(5, 7), // board dimensions
            square_size,
            marker_size, // marker size
            &dictionary,
        )?;
        let detector = CharucoDetector::new_def(&board)?;
        // Collect all image paths

        let mut all_object_points = VectorOfMat::new();
        let mut all_image_points = VectorOfMat::new();

        // Process each image
        let images = images
            .into_iter()
            .map(|img| {
                let img = img?;

                let mut gray = Mat::default();
                opencv::imgproc::cvt_color(
                    &img,
                    &mut gray,
                    imgproc::COLOR_BGR2GRAY,
                    0,
                    core::AlgorithmHint::ALGO_HINT_DEFAULT,
                )?;

                let mut corners = Mat::default();
                let mut ids = Mat::default();

                // Detect markers using ArucoDetector
                detector.detect_board_def(&gray, &mut corners, &mut ids)?;
                if !ids.empty() {
                    let mut object_points = Mat::default();
                    let mut image_points = Mat::default();
                    board.match_image_points(
                        &corners,
                        &ids,
                        &mut object_points,
                        &mut image_points,
                    )?;

                    all_object_points.push(object_points);
                    all_image_points.push(image_points);
                }

                opencv::Result::Ok(img)
            })
            .collect::<opencv::Result<Vec<_>>>()?;

        let image_size = images[0].size()?;
        let mut camera_matrix = Mat::default();
        let mut dist_coeffs = Mat::default();
        let mut rvecs = VectorOfMat::new();
        let mut tvecs = VectorOfMat::new();

        let flags = calib3d::CALIB_FIX_K2
            | calib3d::CALIB_FIX_K3
            | calib3d::CALIB_FIX_K4
            | calib3d::CALIB_FIX_K5
            | calib3d::CALIB_FIX_K6
            | calib3d::CALIB_ZERO_TANGENT_DIST
            | calib3d::CALIB_USE_LU;

        let error = calib3d::calibrate_camera(
            &all_object_points,
            &all_image_points,
            image_size,
            &mut camera_matrix,
            &mut dist_coeffs,
            &mut rvecs,
            &mut tvecs,
            flags,
            core::TermCriteria::default()?,
        )?;
        println!("Calibration error: {}", error);
        println!("Camera matrix:\n{:?}", camera_matrix);
        println!("Distortion coefficients:\n{:?}", dist_coeffs);

        // Tested 22.01. Improved quality
        let camera_matrix = calib3d::get_optimal_new_camera_matrix_def(
            &camera_matrix,
            &dist_coeffs,
            image_size,
            0.0, // alpha=0 to crop all black pixels
        )?;

        Ok(Self {
            detector: Arc::new(detector),
            camera_matrix,
            dist_coeffs,
        })
    }

    pub fn calibrate_extrinsic(
        self,
        distorted: &Mat,
    ) -> Result<ExtrinsicCalibration, opencv::Error> {
        let image_size = distorted.size()?;
        let mut corners = Mat::default();
        let mut ids = Mat::default();
        // Detect markers using ArucoDetector
        self.detector
            .detect_board_def(distorted, &mut corners, &mut ids)?;
        if ids.empty() {
            return Err(opencv::Error::new(
                core::StsObjectNotFound,
                "Nothing detected",
            ));
        }
        let mut object_points = Mat::default();
        let mut image_points = Mat::default();
        self.detector.get_board()?.match_image_points(
            &corners,
            &ids,
            &mut object_points,
            &mut image_points,
        )?;

        // Estimate pose using solvePnP
        let mut rvec = Mat::default();
        let mut tvec = Mat::default();

        calib3d::solve_pnp(
            &object_points,
            &image_points,
            &self.camera_matrix,
            &self.dist_coeffs,
            &mut rvec,
            &mut tvec,
            false,
            calib3d::SOLVEPNP_ITERATIVE,
        )?;

        Ok(ExtrinsicCalibration {
            intrinsic: self,
            rvec,
            tvec,
            image_size,
        })
    }
}

impl ExtrinsicCalibration {
    fn build_world_to_pixel(&self) -> opencv::Result<PixelToWorldLut> {
        let transformer = PixelToWorldTransformer::new(
            &self.intrinsic.camera_matrix,
            &self.rvec,
            &self.tvec,
            &self.intrinsic.dist_coeffs,
        )?;
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
        println!("Lut generation took {:?}", time.elapsed());
        Ok(lut)
    }

    fn debug_points<'a>(
        &'a self,
        lut: &'a PixelToWorldLut,
    ) -> impl Iterator<Item = opencv::Result<(Point2f, Point2f)>> + 'a {
        let iter = (-1..7).flat_map(|x| (-1..9).map(move |y| (x as f32 * 29., y as f32 * 29.)));
        iter.clone().map(|(x, y)| {
            let point_world = Point3f::new(x, y, 0.0);
            let points_world = Vector::from_slice(&[point_world]);
            let projected_points = self.project_points(&points_world)?;
            let coord = projected_points.at::<Point2f>(0)?;

            println!("Pixel position for {points_world:?}: {:?}", coord);
            let pixel_x_rounded = coord.x.round();
            let pixel_y_rounded = coord.y.round();

            if (0..self.image_size.width).contains(&(pixel_x_rounded as i32))
                && (0..self.image_size.height).contains(&(pixel_y_rounded as i32))
            {
                let new_world = lut.get(pixel_x_rounded as _, pixel_y_rounded as _);
                println!("World coordinates: {:?}", new_world);
            } else {
                println!("Not within the image");
            }

            Ok((Point2f::new(x, y), Point2f::new(coord.x, coord.y)))
        })
    }
    fn project_points(&self, points_world: &Vector<Point3f>) -> opencv::Result<Mat> {
        let mut projected_points = Mat::default();

        calib3d::project_points_def(
            &points_world,
            &self.rvec,
            &self.tvec,
            &self.intrinsic.camera_matrix,
            &self.intrinsic.dist_coeffs,
            &mut projected_points,
        )?;

        Ok(projected_points)
    }
}

// In your main function, after getting rvec and tvec from solvePnP:

#[cfg(test)]
mod tests {
    use opencv::core::Point;

    use super::*;

    #[test]
    fn run_charuco() -> opencv::Result<()> {
        let iter = ImageIterable::create_image_iterator("charuco_raw_images");
        let intrinsic = IntrinsicCalibration::create(iter.iter_images().map(|(_, i)| i))?;
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
            let points = extrinsic
                .debug_points(&world_transformer)
                .collect::<opencv::Result<Vec<_>>>()?;

            for (label, mut image) in [("distorded", distorted), ("undistorted", undistorted)] {
                let thickness = 2;
                for (_, pixel) in points.iter() {
                    let pixel_quantisized = Point::new(pixel.x.round() as _, pixel.y.round() as _);
                    imgproc::circle(
                        &mut image,
                        pixel_quantisized,
                        10,
                        core::Scalar::new(255.0, 0.0, 0.0, 1.0),
                        thickness,
                        imgproc::LINE_8,
                        0,
                    )?;
                    imgproc::circle(
                        &mut image,
                        pixel_quantisized,
                        12,
                        core::Scalar::new(0.0, 255.0, 0., 1.0),
                        thickness,
                        imgproc::LINE_8,
                        0,
                    )?;
                }
                let zero =
                    extrinsic.project_points(&Vector::from_elem(Point3f::new(0., 0., 0.), 1))?;
                let zero = zero.at::<Point2f>(0)?;

                imgproc::circle(
                    &mut image,
                    Point::new(zero.x.round() as _, zero.y.round() as _),
                    14,
                    core::Scalar::new(0.0, 0.0, 255.0, 1.0),
                    thickness,
                    imgproc::LINE_8,
                    0,
                )?;

                opencv::imgcodecs::imwrite(
                    &std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                        .join("target")
                        .join(format!("{stem}_{label}.png"))
                        .to_string_lossy()
                        .to_string(),
                    &image,
                    &Vector::new(),
                )?;
            }
        }
        Ok(())
    }
}
