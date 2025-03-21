use std::num::NonZeroU32;

use opencv::{
    calib3d::rodrigues,
    core::{self, Mat, Size_},
    prelude::*,
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

trait PixelToWorld {
    fn convert(x: f32, y: f32) -> (f32, f32);
}

#[derive(Debug, Clone)]
pub struct PixelToWorldLut {
    image_size: Size_<NonZeroU32>,
    data: Box<[[f32; 2]]>,
}

impl PixelToWorldLut {
    pub fn new(transformer: &PixelToWorldTransformer, image_size: Size_<NonZeroU32>) -> Self {
        Self {
            image_size,
            data: (0..image_size.height.get())
                .into_par_iter()
                .flat_map(|y| {
                    (0..image_size.width.get())
                        .into_par_iter()
                        .map(move |x| (x, y))
                })
                .map(|(x, y)| transformer.transform_point(x as f32, y as f32))
                .collect::<Box<[_]>>(),
        }
    }

    #[inline(always)]
    pub fn get(&self, x: u32, y: u32) -> [f32; 2] {
        let width = self.image_size.width.get();
        let height = self.image_size.height.get();

        assert!(x < width, "X coordinate {} exceeds width {}", x, width);
        assert!(y < height, "Y coordinate {} exceeds height {}", y, height);

        // SAFETY: Verified by assert!
        unsafe { *self.data.get_unchecked((y * width + x) as usize) }
    }
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, idx: usize) -> [f32; 2] {
        debug_assert!(
            idx < self.image_size.width.get() as usize * self.image_size.height.get() as usize,
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
    ) -> Result<Self, crate::Error> {
        // Validate matrix sizes at initialization
        if camera_matrix.rows() != 3 || camera_matrix.cols() != 3 {
            return Err(opencv::Error::new(
                core::StsBadArg,
                format!(
                    "Camera matrix must be 3x3, got {}x{}",
                    camera_matrix.rows(),
                    camera_matrix.cols()
                ),
            )
            .into());
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
            )
            .into());
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
            )
            .into());
        }
        // Convert translation vector to array
        let mut tvec_inv = [0.0; 3];
        for ((_, tvec_val), tvec_inv_val) in tvec_inv_mat.iter::<f64>()?.zip(tvec_inv.iter_mut()) {
            *tvec_inv_val = tvec_val;
        }

        // Convert camera matrix inverse to array
        let cam_matrix_inv = camera_matrix.inv_def()?.to_mat()?;
        if cam_matrix_inv.rows() != 3 || cam_matrix_inv.cols() != 3 {
            return Err(
                opencv::Error::new(core::StsBadArg, "Camera matrix inverse must be 3x3").into(),
            );
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
    pub fn transform_point(&self, x: f32, y: f32) -> [f32; 2] {
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

        // Switched 28.01. because i didn't have time to investigate how coordinates are handled (one implementation depends, that 2d/3d are not viewed from other z-direction)
        [wy, wx]
    }
}
