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

pub const DEVICE_TYPE: &str = "engineering-emulation-camera";

pub(super) fn register_services(c: &mut ServiceCollection) {
    c.with::<(Registered<ActorSystem>, Registered<FileServiceBuilder>)>()
        .register_device(DEVICE_TYPE, validator, device);
}

struct DeviceState {
    file_service: FileService<()>,
    actor_system: ActorSystem,
    projector: tokio::sync::watch::Sender<CalibrationResult<PixelToWorldLut>>,
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
    let initial_projector = params.calculate_lut(&file_service).await;
    actor_system
        .register(id)
        .add_handler(DeviceState::update_params)
        .execute(DeviceState {
            file_service,
            actor_system: actor_system.clone(),
            projector: tokio::sync::watch::channel(initial_projector).0,
        })
        .await;

    Ok(())
}

impl DeviceState {
    async fn update_params(
        &mut self,
        UpdateParamsMessage { params }: UpdateParamsMessage<Params>,
    ) -> impl HandlerResult<UpdateParamsMessage<Params>> {
        Ok(())
    }
}

fn create_relative_path() -> &'static RelativeDirectoryPath {
    RelativeDirectoryPath::new("intrinsic").unwrap()
}

#[derive(Debug, Deserialize, Serialize, Clone, Default)]
#[serde(deny_unknown_fields, default)]
pub struct Params {
    extrinsic_base: Option<RelativeFilePath>,
}

impl Params {
    async fn calculate_lut(&self, files: &FileService<()>) -> CalibrationResult<PixelToWorldLut> {
        let intrinsics = files
            .list_files(create_relative_path())
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
            let extrinsic_base_mat = extrinsic_base
                .as_ref()
                .and_then(load_mat)
                .ok_or_else(|| CalibrationError::NotInitialized)??;

            let intrinsic_calib =
                IntrinsicCalibration::create(intrinsics.iter().filter_map(load_mat))?;
            let extrinsic_calib = intrinsic_calib.calibrate_extrinsic(&extrinsic_base_mat)?;
            Ok(extrinsic_calib.build_world_to_pixel()?)
        })
        .await
        .map_err(|_| CalibrationError::NotInitialized)?
    }
}

pub fn create_default_device_config() -> pilatus::DeviceConfig {
    pilatus::DeviceConfig::new_unchecked(DEVICE_TYPE, DEVICE_TYPE, Params::default())
}
