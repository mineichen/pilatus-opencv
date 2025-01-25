use std::num::NonZeroU32;

use opencv::{
    calib3d::rodrigues,
    core::{self, Mat},
    prelude::*,
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

trait PixelToWorld {
    fn convert(x: f32, y: f32) -> (f32, f32);
}

#[derive(Debug, Clone)]
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
    pub fn new(
        camera_matrix: &Mat,
        rvec: &Mat,
        tvec: &Mat,
        dist_coeffs: &Mat,
    ) -> opencv::Result<Self> {
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
    pub fn transform_point(&self, x: f32, y: f32) -> (f32, f32) {
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
