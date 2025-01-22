use std::fs;

use opencv::{
    calib3d::{self},
    core::{self, Mat, Point, Point2f, Point3f, Vector},
    imgproc,
    objdetect::{
        get_predefined_dictionary, CharucoBoard, CharucoDetector, PredefinedDictionaryType,
    },
    prelude::{CharucoDetectorTraitConst, *},
};

mod manual;

type VectorOfMat = Vector<Mat>;
type DynError = Box<dyn std::error::Error>;

fn target_dir(filename: impl AsRef<str>) -> String {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join(filename.as_ref())
        .to_string_lossy()
        .to_string()
}

pub fn charuco() -> Result<(), DynError> {
    // Directory containing calibration images
    let image_dir = "charuco_raw_images";
    let square_size = 29.0; // mm
    let marker_size = 14.0; // mm

    // Create ChArUco board

    let dictionary = get_predefined_dictionary(PredefinedDictionaryType::DICT_5X5_1000)?;

    // Collect all image paths
    let mut paths: Vec<_> = fs::read_dir(image_dir)?
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry
                .path()
                .extension()
                .map(|ext| ext == "jpeg" || ext == "png")
                .unwrap_or(false)
        })
        .map(|entry| entry.path())
        .collect();
    paths.sort_unstable();
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
    let images = paths
        .iter()
        .map(|path| {
            let img =
                opencv::imgcodecs::imread(path.to_str().unwrap(), opencv::imgcodecs::IMREAD_COLOR)?;

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
                board.match_image_points(&corners, &ids, &mut object_points, &mut image_points)?;

                all_object_points.push(object_points);
                all_image_points.push(image_points);
            }

            Result::<_, DynError>::Ok((path.file_stem().unwrap().to_string_lossy(), img))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let image_size = images[0].1.size()?;
    let mut camera_matrix = Mat::default();
    let mut dist_coeffs = Mat::default();
    let mut rvecs = VectorOfMat::new();
    let mut tvecs = VectorOfMat::new();

    let error = calib3d::calibrate_camera(
        &all_object_points,
        &all_image_points,
        image_size,
        &mut camera_matrix,
        &mut dist_coeffs,
        &mut rvecs,
        &mut tvecs,
        calib3d::CALIB_RATIONAL_MODEL,
        core::TermCriteria::default()?,
    )?;
    println!("Calibration error: {}", error);
    println!("Camera matrix:\n{:?}", camera_matrix);
    println!("Distortion coefficients:\n{:?}", dist_coeffs);

    // Calibrate camera

    let mut corners = Mat::default();
    let mut ids = Mat::default();

    // Tested 22.01. Improted quality
    let camera_matrix = calib3d::get_optimal_new_camera_matrix_def(
        &camera_matrix,
        &dist_coeffs,
        image_size,
        0.0, // alpha=0 to crop all black pixels
    )?;

    for (img_filename_stem, distorted) in images {
        // Detect markers using ArucoDetector
        detector.detect_board_def(&distorted, &mut corners, &mut ids)?;
        if !ids.empty() {
            let mut object_points = Mat::default();
            let mut image_points = Mat::default();
            board.match_image_points(&corners, &ids, &mut object_points, &mut image_points)?;

            // Estimate pose using solvePnP
            let mut rvec = Mat::default();
            let mut tvec = Mat::default();

            calib3d::solve_pnp(
                &object_points,
                &image_points,
                &camera_matrix,
                &dist_coeffs,
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
                &camera_matrix,
                &dist_coeffs,
                &Mat::default(),
                &camera_matrix, // Use same camera matrix for simplicity
                image_size,
                core::CV_32FC1,
                &mut map1,
                &mut map2,
            )?;

            // Remap the image
            imgproc::remap(
                &distorted,
                &mut undistorted,
                &map1,
                &map2,
                imgproc::INTER_LINEAR,
                core::BORDER_CONSTANT,
                core::Scalar::default(),
            )?;

            let coords = (-1..7)
                .flat_map(|x| (-1..9).map(move |y| (x as f32 * 29., y as f32 * 29.)))
                .map(|(x, y)| {
                    let point = Vector::from_slice(&[Point3f::new(x, y, 0.0)]);
                    let mut projected_points = Mat::default();

                    calib3d::project_points_def(
                        &point,
                        &rvec,
                        &tvec,
                        &camera_matrix,
                        &dist_coeffs,
                        &mut projected_points,
                    )?;
                    let coord = projected_points.at::<Point2f>(0)?;
                    Result::<_, DynError>::Ok(Point::new(coord.x as i32, coord.y as i32))
                })
                .collect::<Result<Vec<_>, _>>()?;
            for (label, mut image) in [("distorded", distorted), ("undistorted", undistorted)] {
                let thickness = 2;
                for coord in coords.iter() {
                    println!("Pixel position: {:?}", coord);

                    imgproc::circle(
                        &mut image,
                        *coord,
                        10,
                        core::Scalar::new(255.0, 0.0, 0.0, 1.0),
                        thickness,
                        imgproc::LINE_8,
                        0,
                    )?;
                    imgproc::circle(
                        &mut image,
                        *coord,
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
                    &image,
                    &Vector::new(),
                )?;
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_charuco() {
        charuco().unwrap();
    }
}
