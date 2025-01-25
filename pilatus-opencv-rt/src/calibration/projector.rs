use futures::{stream::BoxStream, StreamExt};
use pilatus::{
    device::{ActorMessage, ActorResult},
    Name, RelativeDirectoryPath,
};

use crate::calibration::{CalibrationResult, PixelToWorldLut};

use super::DeviceState;

pub(super) struct StreamProjectorMessage;

impl ActorMessage for StreamProjectorMessage {
    type Output = BoxStream<'static, CalibrationResult<PixelToWorldLut>>;
    type Error = std::convert::Infallible;
}

impl DeviceState {
    pub(super) async fn stream_projector(
        &mut self,
        _msg: StreamProjectorMessage,
    ) -> ActorResult<StreamProjectorMessage> {
        Ok(tokio_stream::wrappers::WatchStream::new(self.projector.subscribe()).boxed())
    }
}
