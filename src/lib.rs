use std::{fs, num::NonZeroU32};

use opencv::{
    calib3d::{self, rodrigues},
    core::{self, Mat, Point, Point2f, Point3f, Vector},
    imgproc,
    objdetect::{
        get_predefined_dictionary, CharucoBoard, CharucoDetector, PredefinedDictionaryType,
    },
    prelude::CharucoDetectorTraitConst,
    prelude::*,
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

mod image;
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

    // Calibrate camera

    let mut corners = Mat::default();
    let mut ids = Mat::default();

    // Tested 22.01. Improved quality
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
                core::Scalar::all(0.0),
            )?;
            let transformer =
                PixelToWorldTransformer::new(&camera_matrix, &rvec, &tvec, &dist_coeffs)?;
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
                        &camera_matrix,
                        &dist_coeffs,
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
                    // let world_coords = pixel_to_world(
                    //     &camera_matrix,
                    //     &rvec,
                    //     &tvec,
                    //     coord.x as _, // Your pixel x coordinate
                    //     coord.y as _, // Your pixel y coordinate
                    // )?;
                    //

                    Result::<_, DynError>::Ok(Point::new(coord.x as i32, coord.y as i32))
                })
                .collect::<Result<Vec<_>, _>>()?;
            for (label, mut image) in [("distorded", distorted), ("undistorted", undistorted)] {
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
                    &image,
                    &Vector::new(),
                )?;
            }
        }
    }

    Ok(())
}
#[derive(Debug)]
pub struct PixelToWorldLut {
    width: NonZeroU32,
    height: NonZeroU32,
    data: Box<[(f32, f32)]>,
}

impl PixelToWorldLut {
    pub fn new(
        transformer: &PixelToWorldTransformer,
        width: NonZeroU32,
        height: NonZeroU32,
    ) -> Self {
        Self {
            width,
            height,
            data: (0..height.get())
                .into_par_iter()
                .flat_map(|y| (0..width.get()).into_par_iter().map(move |x| (x, y)))
                .map(|(x, y)| transformer.transform_point(x as f32, y as f32))
                .collect::<Box<[_]>>(),
        }
    }

    #[inline(always)]
    pub fn get(&self, x: u32, y: u32) -> (f32, f32) {
        let width = self.width.get();
        let height = self.height.get();

        assert!(x < width, "X coordinate {} exceeds width {}", x, width);
        assert!(y < height, "Y coordinate {} exceeds height {}", y, height);

        // SAFETY: Verified by assert!
        unsafe { *self.data.get_unchecked((y * width + x) as usize) }
    }
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, idx: usize) -> (f32, f32) {
        debug_assert!(
            idx < self.width.get() as usize * self.height.get() as usize,
            "Invalid access"
        );

        *self.data.get_unchecked(idx)
    }
}

pub struct PixelToWorldTransformer {
    rot_mat_inv: [[f64; 3]; 3],
    tvec_inv: [f64; 3],
    cam_matrix_inv: [[f64; 3]; 3],
    kappa: f64,
}

impl PixelToWorldTransformer {
    fn new(camera_matrix: &Mat, rvec: &Mat, tvec: &Mat, dist_coeffs: &Mat) -> opencv::Result<Self> {
        // Validate matrix sizes at initialization
        if camera_matrix.rows() != 3 || camera_matrix.cols() != 3 {
            return Err(opencv::Error::new(
                core::StsBadArg,
                format!(
                    "Camera matrix must be 3x3, got {}x{}",
                    camera_matrix.rows(),
                    camera_matrix.cols()
                ),
            ));
        }

        // Compute rotation matrix inverse
        let mut rot_mat = Mat::default();
        rodrigues(rvec, &mut rot_mat, &mut Mat::default())?;
        if rot_mat.rows() != 3 || rot_mat.cols() != 3 {
            return Err(opencv::Error::new(
                core::StsBadArg,
                format!(
                    "Rotation matrix must be 3x3, got {}x{}",
                    camera_matrix.rows(),
                    camera_matrix.cols()
                ),
            ));
        }

        let rot_mat_inv_cv = rot_mat.inv_def()?.to_mat()?;
        // Convert rotation matrix to array
        let mut rot_mat_inv = [[0.0; 3]; 3];
        for row in 0..3 {
            for col in 0..3 {
                rot_mat_inv[row][col] = *rot_mat_inv_cv.at_2d::<f64>(row as i32, col as i32)?;
            }
        }

        // Compute translation vector inverse
        let mut tvec_inv_mat = Mat::default();
        core::gemm(
            &rot_mat.t()?,
            tvec,
            -1.0,
            &Mat::default(),
            0.0,
            &mut tvec_inv_mat,
            0,
        )?;
        if tvec_inv_mat.rows() != 3 || tvec_inv_mat.cols() != 1 {
            return Err(opencv::Error::new(
                core::StsBadArg,
                format!(
                    "Translation vector must be 3x1, got {}x{}",
                    camera_matrix.rows(),
                    camera_matrix.cols()
                ),
            ));
        }

        // Convert translation vector to array
        let mut tvec_inv = [0.0; 3];
        for i in 0..3 {
            tvec_inv[i] = *tvec_inv_mat.at::<f64>(i as i32)?;
        }

        // Convert camera matrix inverse to array
        let cam_matrix_inv = camera_matrix.inv_def()?.to_mat()?;
        if cam_matrix_inv.rows() != 3 || cam_matrix_inv.cols() != 3 {
            return Err(opencv::Error::new(
                core::StsBadArg,
                "Camera matrix inverse must be 3x3",
            ));
        }

        let mut cam_matrix_inv_arr = [[0.0; 3]; 3];
        for row in 0..3 {
            for col in 0..3 {
                cam_matrix_inv_arr[row][col] =
                    *cam_matrix_inv.at_2d::<f64>(row as i32, col as i32)?;
            }
        }
        let kappa = if dist_coeffs.rows() >= 1 && dist_coeffs.cols() >= 1 {
            for idx in 1..(dist_coeffs.cols() * dist_coeffs.rows()) {
                assert_eq!(0., *dist_coeffs.at::<f64>(idx).unwrap());
            }
            0.0
            //-*dist_coeffs.at::<f64>(0)?
        } else {
            0.0
        };

        Ok(Self {
            rot_mat_inv,
            tvec_inv,
            cam_matrix_inv: cam_matrix_inv_arr,
            kappa,
        })
    }

    #[inline]
    fn transform_point(&self, x: f32, y: f32) -> (f32, f32) {
        let x = x as f64;
        let y = y as f64;

        let k = self.kappa; // Your κ parameter
        let r2 = x * x + y * y;
        let scale = 1.0 / (1.0 + k * r2);
        let xu = x * scale;
        let yu = y * scale;

        // Camera coordinates using array accesses
        let ray_camera_x = self.cam_matrix_inv[0][0] * xu + self.cam_matrix_inv[0][2];
        let ray_camera_y = self.cam_matrix_inv[1][1] * yu + self.cam_matrix_inv[1][2];
        let ray_camera_z = 1.0;

        // World coordinates using precomputed arrays
        let rx = self.rot_mat_inv[0][0] * ray_camera_x
            + self.rot_mat_inv[0][1] * ray_camera_y
            + self.rot_mat_inv[0][2] * ray_camera_z;

        let ry = self.rot_mat_inv[1][0] * ray_camera_x
            + self.rot_mat_inv[1][1] * ray_camera_y
            + self.rot_mat_inv[1][2] * ray_camera_z;

        let rz = self.rot_mat_inv[2][0] * ray_camera_x
            + self.rot_mat_inv[2][1] * ray_camera_y
            + self.rot_mat_inv[2][2] * ray_camera_z;

        // Plane intersection
        let t = -self.tvec_inv[2] / rz;
        let wx = (self.tvec_inv[0] + rx * t) as f32;
        let wy = (self.tvec_inv[1] + ry * t) as f32;

        (wx, wy)
    }
}

// In your main function, after getting rvec and tvec from solvePnP:

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_charuco() {
        charuco().unwrap();
    }
}
