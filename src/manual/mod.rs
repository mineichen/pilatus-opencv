use std::ops::Deref;

use opencv::{
    calib3d::{self, CALIB_CB_ADAPTIVE_THRESH},
    core::{self as cvcore, Mat, MatTraitConst, Point3f, Vector},
    imgcodecs, Result,
};
use pilatus_engineering::image::LumaImage;

mod image;

pub fn show_image() -> Result<(), Box<dyn std::error::Error>> {
    let mut all_image_points: Vector<Mat> = Vector::new();
    let mut all_world_points: Vector<Mat> = Vector::new();

    let image = LumaImage::new_vec(
        vec![0, 255, 255, 0],
        2.try_into().unwrap(),
        2.try_into().unwrap(),
    );

    let cv_image: image::BorrowImage = (&image).try_into()?;
    imgcodecs::imwrite(
        &super::target_dir("square.png"),
        cv_image.deref(),
        &Default::default(),
    )?;

    let mut src: opencv::core::Mat = imgcodecs::imread("bad_board.png", imgcodecs::IMREAD_COLOR)?;

    let mut image_points = opencv::core::Mat::default();
    let pattern_size = opencv::core::Size::new(9, 14);
    let found = opencv::calib3d::find_chessboard_corners(
        &src,
        pattern_size,
        &mut image_points,
        CALIB_CB_ADAPTIVE_THRESH,
    )?;
    println!("Found pattern {found:?}");
    let mut labled = src.clone();
    //let clone = Mat::from(src.deref());
    opencv::calib3d::draw_chessboard_corners(&mut labled, pattern_size, &image_points, found)?;
    imgcodecs::imwrite(
        &super::target_dir("labled_chessboard.png"),
        &labled,
        &Default::default(),
    )?;
    all_image_points.push(image_points);

    // const FRACTION_OFFSET: i32 = 1;
    // opencv::imgproc::circle(
    //     &mut src,
    //     opencv::core::Point::new(407 << FRACTION_OFFSET, 349 << FRACTION_OFFSET),
    //     15,
    //     opencv::core::Scalar::from((255.0, 0., 0.)),
    //     1,
    //     opencv::imgproc::LINE_AA,
    //     FRACTION_OFFSET,
    // )?;

    // for p in all_image_points.iter() {
    //     println!("All_image_points {p:?}");
    // }

    let mut camera_matrix = Mat::default();
    let mut dist_coeffs = Mat::default();
    let mut rvecs = Mat::default();
    let mut tvecs = Mat::default();

    let world_points: Vec<Point3f> = (0..pattern_size.height)
        .flat_map(|y| (0..pattern_size.width).map(move |x| Point3f::new(x as f32, y as f32, 0.0)))
        .collect();
    let world_points_mat = Mat::from_slice(&world_points)?;
    all_world_points.push(world_points_mat.try_clone()?);

    let clib: f64 = calib3d::calibrate_camera(
        &all_world_points,
        &all_image_points,
        src.size()?,
        &mut camera_matrix,
        &mut dist_coeffs,
        &mut rvecs,
        &mut tvecs,
        calib3d::CALIB_FIX_ASPECT_RATIO,
        cvcore::TermCriteria::default()?,
    )?;
    println!("Camera Matrix: {:?}", camera_matrix);
    println!("Distortion Coefficients: {:?}", dist_coeffs);

    let undistorted = undistort_image(&src, &camera_matrix, &dist_coeffs)?;
    imgcodecs::imwrite("undistorted_image.png", &undistorted, &Vector::new())?;

    //highgui::imshow("hello opencv!", &src)?;
    Ok(())
}

fn undistort_image(src: &Mat, camera_matrix: &Mat, dist_coeffs: &Mat) -> Result<Mat> {
    let mut undistorted = Mat::default();
    calib3d::undistort(
        src,
        &mut undistorted,
        camera_matrix,
        dist_coeffs,
        &camera_matrix,
    )?;

    Ok(undistorted)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        show_image().unwrap()
    }
}
