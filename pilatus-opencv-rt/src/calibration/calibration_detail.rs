use opencv::core::VectorToVec;
use pilatus::device::{ActorResult, ActorSystem, DynamicIdentifier};
use pilatus_axum::{
    extract::{InjectRegistered, Query},
    DeviceResponse, IntoResponse,
};
use pilatus_opencv::calibration::{CalibrationDetailMessage, NoCalibrationDetailsError};

use super::DeviceState;

impl DeviceState {
    pub(super) async fn get_calibration_details(
        &mut self,
        _msg: CalibrationDetailMessage,
    ) -> ActorResult<CalibrationDetailMessage> {
        self.artifacts
            .calibration_details
            .as_ref()
            .map(|x| x.to_vec())
            .ok_or_else(|| NoCalibrationDetailsError.into())
    }
}

pub async fn web_handler(
    InjectRegistered(system): InjectRegistered<ActorSystem>,
    Query(id): Query<DynamicIdentifier>,
) -> impl IntoResponse {
    DeviceResponse::from(
        system
            .ask(id, CalibrationDetailMessage::default())
            .await
            .map(|d| ([("content-type", "image/png")], d)),
    )
}
