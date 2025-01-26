use std::{path::Path, sync::Arc};

use opencv::{
    calib3d::{self},
    core::{self, Mat, Vector},
    imgproc,
    objdetect::{
        get_predefined_dictionary, CharucoBoard, CharucoDetector, PredefinedDictionaryType,
    },
    prelude::{CharucoDetectorTraitConst, *},
};
use tracing::{debug, trace};

use super::{ExtrinsicCalibration, Undistorter};

type VectorOfMat = Vector<Mat>;

#[derive(Clone)]
pub struct IntrinsicCalibration {
    detector: Arc<CharucoDetector>,
    camera_matrix: Mat,
    dist_coeffs: Mat,
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
    pub fn camera_matrix(&self) -> &Mat {
        &self.camera_matrix
    }

    pub fn dist_coeffs(&self) -> &Mat {
        &self.dist_coeffs
    }

    pub fn create(
        images: impl IntoIterator<Item = opencv::Result<Mat>>,
    ) -> Result<IntrinsicCalibration, opencv::Error> {
        // Directory containing calibration images

        let square_size = 40.0; // mm
        let marker_size = 20.0; // mm

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

        let image_size = images
            .get(0)
            .ok_or_else(|| {
                opencv::Error::new(
                    core::StsBadArg,
                    "Require at least one input image to perform calibration",
                )
            })?
            .size()?;
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
        debug!("Calibration error: {}", error);
        trace!("Camera matrix:\n{:?}", camera_matrix);
        trace!("Distortion coefficients:\n{:?}", dist_coeffs);

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
