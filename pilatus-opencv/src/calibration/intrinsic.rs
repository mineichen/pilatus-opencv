use std::sync::Arc;

use opencv::{
    calib3d::{self},
    core::{self, Mat, Point2f, Point3f, Size_, Vector},
    imgproc::{self, THRESH_BINARY},
    objdetect::{
        get_predefined_dictionary, CharucoBoard, CharucoDetector, PredefinedDictionaryType,
    },
    prelude::{CharucoDetectorTraitConst, *},
};
use serde::{Deserialize, Serialize};
use tracing::{debug, trace};

use crate::calibration::CalibrationResult;

use super::{CalibrationError, ExtrinsicCalibration, Undistorter};

type VectorOfMat = Vector<Mat>;

#[derive(Clone)]
pub struct IntrinsicCalibration {
    detector: Arc<CharucoDetector>,
    // Threshold to generate threshold-image for marker detection
    threshold: u8,
    camera_matrix: Mat,
    dist_coeffs: Mat,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct IntrinsicCalibrationSettings {
    square_size_mm: f32,
    marker_size_mm: f32,
    threshold: u8,
}

impl Default for IntrinsicCalibrationSettings {
    fn default() -> Self {
        Self {
            square_size_mm: 40.0,
            marker_size_mm: 20.0,
            threshold: 145,
        }
    }
}

impl IntrinsicCalibration {
    pub fn create_undistorter(
        &self,
        image_size: core::Size_<i32>,
    ) -> Result<Undistorter, crate::Error> {
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

    pub fn board(&self) -> Result<CharucoBoard, crate::Error> {
        self.detector.get_board().map_err(Into::into)
    }

    pub fn dist_coeffs(&self) -> &Mat {
        &self.dist_coeffs
    }

    pub fn create(
        images: impl IntoIterator<Item = Result<Mat, crate::Error>>,
        settings: IntrinsicCalibrationSettings,
    ) -> Result<IntrinsicCalibration, CalibrationError> {
        // Directory containing calibration images

        // Create ChArUco board

        let dictionary = get_predefined_dictionary(PredefinedDictionaryType::DICT_5X5_1000)?;
        let board = CharucoBoard::new_def(
            opencv::core::Size::new(5, 7), // board dimensions
            settings.square_size_mm,
            settings.marker_size_mm, // marker size
            &dictionary,
        )?;

        let detector = CharucoDetector::new_def(&board)?;
        // Collect all image paths

        let mut all_object_points = VectorOfMat::new();
        let mut all_image_points = VectorOfMat::new();

        // Process each image
        let sizes = images
            .into_iter()
            .map(|img| {
                let img = img?;
                let (object_points, image_points) =
                    Self::match_board(&detector, &img, settings.threshold)?;
                all_object_points.push(object_points);
                all_image_points.push(image_points);

                CalibrationResult::Ok(img.size()?)
            })
            .collect::<CalibrationResult<Vec<_>>>()?;

        let image_size = *sizes.get(0).ok_or_else(|| {
            opencv::Error::new(
                core::StsBadArg,
                "Require at least one input image to perform calibration",
            )
        })?;
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

        trace!("Before cali_camera");
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
            threshold: settings.threshold,
        })
    }

    // ObjectPoints/ImagePoints
    fn match_board(
        detector: &CharucoDetector,
        input: &Mat,
        threshold: u8,
    ) -> CalibrationResult<(Mat, Mat)> {
        //println!("before board detect");
        let mut gray = Mat::default();
        opencv::imgproc::cvt_color(
            &input,
            &mut gray,
            imgproc::COLOR_BGR2GRAY,
            0,
            core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;
        //println!("after gray");
        let mut binary = Mat::default();
        opencv::imgproc::threshold(&gray, &mut binary, threshold as f64, 255., THRESH_BINARY)?;

        //println!("after thres");
        let mut corners = Mat::default();
        let mut ids = Mat::default();

        // Detect markers using ArucoDetector
        detector.detect_board_def(&binary, &mut corners, &mut ids)?;

        let board = detector.get_board()?;
        let num_detected = corners.rows();
        if num_detected < 20 {
            let points = if num_detected != 0 {
                trace!("after board {}", corners.rows());
                let points: Vec<_> = corners.iter::<Point2f>()?.map(|(_, p)| p).collect();
                for p in points.iter() {
                    super::draw_inforced_circle(&mut binary, p)?;
                }
                points
            } else {
                Vec::new()
            };

            debug!("Corners in calibration image: {corners:?}");
            return Err(CalibrationError::NotEnoughPoints {
                image: binary,
                points,
            });
        }
        let mut object_points = Mat::default();
        let mut image_points = Mat::default();

        trace!("before match_image_points");
        board.match_image_points(&corners, &ids, &mut object_points, &mut image_points)?;
        trace!("after match_image_points");

        if image_points.rows() < 4 {
            let points: Vec<_> = image_points.iter::<Point2f>()?.map(|(_, p)| p).collect();
            for p in points.iter() {
                super::draw_inforced_circle(&mut binary, p)?;
            }
            debug!("Corners in reference image: {corners:?}");
            return Err(CalibrationError::NotEnoughPoints {
                image: binary,
                points,
            });
        }
        Ok((object_points, image_points))
    }

    pub fn calibrate_extrinsic_points(
        self,
        object_points: &Mat,
        image_points: &Mat,
        signed_image_size: Size_<i32>,
    ) -> CalibrationResult<ExtrinsicCalibration> {
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

        let image_size = Size_::new(
            u32::try_from(signed_image_size.width)
                .and_then(|signed| signed.try_into())
                .map_err(opencv::Error::from)?,
            u32::try_from(signed_image_size.height)
                .and_then(|signed| signed.try_into())
                .map_err(opencv::Error::from)?,
        );
        dbg!(&rvec);
        dbg!(&tvec);

        Ok(ExtrinsicCalibration {
            intrinsic: self,
            position: super::CameraPosition { rvec, tvec },
            image_size,
        })
    }

    pub fn calibrate_extrinsic(self, distorted: &Mat) -> CalibrationResult<ExtrinsicCalibration> {
        let signed_image_size = distorted.size()?;
        let (object_points, image_points) =
            Self::match_board(&self.detector, &distorted, self.threshold)?;

        // for x in object_points.iter::<Point3f>()?.map(|(_, x)| x) {
        //     trace!("World: {x:?}");
        // }

        tracing::info!("Before writing image points: {image_points:?}");
        // for x in image_points.iter::<Point2f>()?.map(|(_, x)| x) {
        //     trace!("Image: {x:?}");
        // }

        self.calibrate_extrinsic_points(&object_points, &image_points, signed_image_size)
    }
}
