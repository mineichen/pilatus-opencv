use futures::StreamExt;
use pilatus::device::ActorResult;
use pilatus_opencv::calibration::StreamProjectorMessage;

use super::DeviceState;

impl DeviceState {
    pub(super) async fn stream_projector(
        &mut self,
        _msg: StreamProjectorMessage,
    ) -> ActorResult<StreamProjectorMessage> {
        Ok(tokio_stream::wrappers::WatchStream::new(self.artifacts.lut.subscribe()).boxed())
    }
}
