use std::path::{Path, PathBuf};

use opencv::{
    calib3d::{self},
    core::{self, Mat, Point, Point2f, Point3f, Vector},
    imgproc,
    objdetect::{
        get_predefined_dictionary, CharucoBoard, CharucoDetector, PredefinedDictionaryType,
    },
    prelude::CharucoDetectorTraitConst,
    prelude::*,
};

mod pixel_to_world;

pub use pixel_to_world::*;

type VectorOfMat = Vector<Mat>;
type DynError = Box<dyn std::error::Error>;

fn target_dir(filename: impl AsRef<str>) -> String {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join(filename.as_ref())
        .to_string_lossy()
        .to_string()
}

pub enum CalibrationError {}

pub struct IntrinsicCalibration {
    images: Vec<(String, Mat)>,
    detector: CharucoDetector,
    camera_matrix: Mat,
    dist_coeffs: Mat,
}

pub struct ExtrinsicCalibration {}

impl IntrinsicCalibration {
    pub fn from_dir(path: impl AsRef<Path>) -> Result<IntrinsicCalibration, DynError> {
        let mut paths: Vec<_> = std::fs::read_dir(path)
            .unwrap()
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry
                    .path()
                    .extension()
                    .map(|ext| ext == "jpg" || ext == "jpeg" || ext == "png")
                    .unwrap_or(false)
            })
            .map(|entry| entry.path())
            .collect();
        paths.sort_unstable();
        Self::create(paths).map_err(Into::into)
    }

    pub fn create(
        images: impl IntoIterator<Item = PathBuf>,
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
            .map(|path| {
                let img = opencv::imgcodecs::imread(
                    path.to_str().unwrap(),
                    opencv::imgcodecs::IMREAD_COLOR,
                )?;

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

                opencv::Result::Ok((path.file_stem().unwrap().to_string_lossy().to_string(), img))
            })
            .collect::<opencv::Result<Vec<_>>>()?;

        let image_size = images[0].1.size()?;
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
            detector,
            images,
            camera_matrix,
            dist_coeffs,
        })
    }

    pub fn calibrate_extrinsic(&mut self) -> Result<ExtrinsicCalibration, opencv::Error> {
        for (img_filename_stem, distorted) in self.images.iter_mut() {
            let image_size = distorted.size()?;
            let mut corners = Mat::default();
            let mut ids = Mat::default();
            // Detect markers using ArucoDetector
            self.detector
                .detect_board_def(distorted, &mut corners, &mut ids)?;
            if !ids.empty() {
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

                //     let mut undistorted = Mat::default();
                //     calib3d::undistort_def(&last, &mut undistorted, &camera_matrix, &dist_coeffs)?;
                //    let mut last = undistorted;

                let mut undistorted = Mat::default();
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

                // Remap the image
                imgproc::remap(
                    distorted,
                    &mut undistorted,
                    &map1,
                    &map2,
                    imgproc::INTER_LINEAR,
                    core::BORDER_CONSTANT,
                    core::Scalar::all(0.0),
                )?;
                let transformer = PixelToWorldTransformer::new(
                    &self.camera_matrix,
                    &rvec,
                    &tvec,
                    &self.dist_coeffs,
                )?;
                let time = std::time::Instant::now();
                let lut = PixelToWorldLut::new(
                    &transformer,
                    (image_size.width as u32)
                        .try_into()
                        .expect("Images are always wider 0"),
                    (image_size.height as u32)
                        .try_into()
                        .expect("Images are always higher 0"),
                );
                println!("Lut generation took {:?}", time.elapsed());
                let coords = (-1..7)
                    .flat_map(|x| (-1..9).map(move |y| (x as f32 * 29., y as f32 * 29.)))
                    .map(|(x, y)| {
                        let point = Vector::from_slice(&[Point3f::new(x, y, 0.0)]);
                        let mut projected_points = Mat::default();

                        calib3d::project_points_def(
                            &point,
                            &rvec,
                            &tvec,
                            &self.camera_matrix,
                            &self.dist_coeffs,
                            &mut projected_points,
                        )?;

                        let coord = projected_points.at::<Point2f>(0)?;

                        println!("Pixel position for {point:?}: {:?}", coord);
                        let pixel_x_rounded = coord.x.round();
                        let pixel_y_rounded = coord.y.round();

                        if (0..image_size.width).contains(&(pixel_x_rounded as i32))
                            && (0..image_size.height).contains(&(pixel_y_rounded as i32))
                        {
                            let new_world = lut.get(pixel_x_rounded as _, pixel_y_rounded as _);
                            println!("World coordinates: {:?}", new_world);
                        } else {
                            println!("Not within the image");
                        }

                        Result::<_, opencv::Error>::Ok(Point::new(coord.x as i32, coord.y as i32))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                for (label, mut image) in
                    [("distorded", distorted), ("undistorted", &mut undistorted)]
                {
                    let thickness = 2;
                    for pixel in coords.iter() {
                        imgproc::circle(
                            &mut image,
                            *pixel,
                            10,
                            core::Scalar::new(255.0, 0.0, 0.0, 1.0),
                            thickness,
                            imgproc::LINE_8,
                            0,
                        )?;
                        imgproc::circle(
                            &mut image,
                            *pixel,
                            12,
                            core::Scalar::new(0.0, 255.0, 0., 1.0),
                            thickness,
                            imgproc::LINE_8,
                            0,
                        )?;
                    }
                    imgproc::circle(
                        &mut image,
                        *coords.get(0).unwrap(),
                        14,
                        core::Scalar::new(0.0, 0.0, 255.0, 1.0),
                        thickness,
                        imgproc::LINE_8,
                        0,
                    )?;
                    opencv::imgcodecs::imwrite(
                        &target_dir(format!("{img_filename_stem}_{label}.png")),
                        image,
                        &Vector::new(),
                    )?;
                }
            }
        }
        Ok(ExtrinsicCalibration {})
    }
}

// In your main function, after getting rvec and tvec from solvePnP:

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_charuco() {
        let mut intrinsic = IntrinsicCalibration::from_dir("charuco_raw_images").unwrap();
        intrinsic.calibrate_extrinsic().unwrap();
    }
}
