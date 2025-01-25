use std::path::PathBuf;

use minfac::{Registered, ServiceCollection};
use pilatus::device::HandlerResult;
use pilatus::{
    device::{ActorSystem, DeviceContext, DeviceResult, DeviceValidationContext},
    prelude::*,
    UpdateParamsMessage, UpdateParamsMessageError,
};
use pilatus::{FileService, FileServiceBuilder, RelativeDirectoryPath, RelativeFilePath};
use serde::{Deserialize, Serialize};

use pilatus_opencv::calibration::{
    CalibrationError, CalibrationResult, IntrinsicCalibration, PixelToWorldLut,
};

mod projector;

pub const DEVICE_TYPE: &str = "opencv-calibration";

pub(super) fn register_services(c: &mut ServiceCollection) {
    c.with::<(Registered<ActorSystem>, Registered<FileServiceBuilder>)>()
        .register_device(DEVICE_TYPE, validator, device);
}

pub struct Artifacts {
    calibration_details: Option<Vec<u8>>,
    lut: tokio::sync::watch::Sender<CalibrationResult<PixelToWorldLut>>,
    _keep_sender_open: tokio::sync::watch::Receiver<CalibrationResult<PixelToWorldLut>>,
}

impl Artifacts {
    fn new(r: CalibrationResult<(Vec<u8>, PixelToWorldLut)>) -> Self {
        let (calibration_details, lut) = Self::split_raw(r);
        let (lut, _keep_sender_open) = tokio::sync::watch::channel(lut);
        Self {
            calibration_details,
            lut,
            _keep_sender_open,
        }
    }

    fn update(&mut self, r: CalibrationResult<(Vec<u8>, PixelToWorldLut)>) {
        let (calibration_details, lut) = Self::split_raw(r);
        self.calibration_details = calibration_details;
        self.lut.send(lut).expect("State keeps the channel open");
    }
    fn split_raw(
        r: CalibrationResult<(Vec<u8>, PixelToWorldLut)>,
    ) -> (Option<Vec<u8>>, CalibrationResult<PixelToWorldLut>) {
        let mut calibration_details = None;
        let initial_projector = r.map(|(details, x)| {
            calibration_details = Some(details);
            x
        });
        (calibration_details, initial_projector)
    }
}

struct DeviceState {
    file_service: FileService<()>,
    actor_system: ActorSystem,
    artifacts: Artifacts,
}

async fn validator(ctx: DeviceValidationContext<'_>) -> Result<Params, UpdateParamsMessageError> {
    ctx.params_as::<Params>()
}

async fn device(
    ctx: DeviceContext,
    params: Params,
    (actor_system, file_service_builder): (ActorSystem, FileServiceBuilder),
) -> DeviceResult {
    let id = ctx.id;
    let file_service = file_service_builder.build(id);
    let r = params.calculate_lut(&file_service).await;

    actor_system
        .register(id)
        .add_handler(DeviceState::update_params)
        .add_handler(DeviceState::stream_projector)
        .execute(DeviceState {
            file_service,
            actor_system: actor_system.clone(),
            artifacts: Artifacts::new(r),
        })
        .await;

    Ok(())
}

impl DeviceState {
    async fn update_params(
        &mut self,
        UpdateParamsMessage { params }: UpdateParamsMessage<Params>,
    ) -> impl HandlerResult<UpdateParamsMessage<Params>> {
        self.artifacts
            .update(params.calculate_lut(&self.file_service).await);
        Ok(())
    }
}

fn intrinsic_path() -> &'static RelativeDirectoryPath {
    RelativeDirectoryPath::new("intrinsic").unwrap()
}

#[derive(Debug, Deserialize, Serialize, Clone, Default)]
#[serde(deny_unknown_fields, default)]
pub struct Params {
    extrinsic_base: Option<RelativeFilePath>,
}

impl Params {
    async fn calculate_lut(
        &self,
        files: &FileService<()>,
    ) -> CalibrationResult<(Vec<u8>, PixelToWorldLut)> {
        let intrinsics = files
            .list_files(intrinsic_path())
            .await
            .map_err(|_| CalibrationError::NotInitialized)?
            .into_iter()
            .map(|p| files.get_filepath(&p))
            .collect::<Vec<_>>();
        let extrinsic_base = self.extrinsic_base.as_ref().map(|p| files.get_filepath(p));

        tokio::task::spawn_blocking(move || {
            let load_mat = |p: &PathBuf| {
                Some(opencv::imgcodecs::imread(
                    p.to_str()?,
                    opencv::imgcodecs::IMREAD_COLOR,
                ))
            };
            let mut extrinsic_base_mat = extrinsic_base
                .as_ref()
                .and_then(load_mat)
                .ok_or_else(|| CalibrationError::NotInitialized)??;

            let intrinsic_calib =
                IntrinsicCalibration::create(intrinsics.iter().filter_map(load_mat))?;
            let extrinsic_calib = intrinsic_calib.calibrate_extrinsic(&extrinsic_base_mat)?;
            let lut = extrinsic_calib.build_world_to_pixel()?;
            extrinsic_calib.draw_debug_points(&mut extrinsic_base_mat, &lut)?;

            Ok((Vec::new(), lut))
        })
        .await
        .map_err(|_| CalibrationError::NotInitialized)?
    }
}

pub fn create_default_device_config() -> pilatus::DeviceConfig {
    pilatus::DeviceConfig::new_unchecked(DEVICE_TYPE, DEVICE_TYPE, Params::default())
}
