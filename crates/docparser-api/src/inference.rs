//! Bounded queue + dedicated worker thread for CPU-bound document parsing.

use std::sync::mpsc::{SyncSender, sync_channel};
use std::thread::{self, JoinHandle};

use docparser_pipeline::{DocumentParseResult, DocumentPipeline, PipelineError};
use image::ImageFormat;
use tokio::sync::oneshot;

pub struct InferencePool {
    tx: SyncSender<InferenceJob>,
    _worker: JoinHandle<()>,
}

struct InferenceJob {
    bytes: Vec<u8>,
    format: ImageFormat,
    filename: Option<String>,
    reply: oneshot::Sender<Result<DocumentParseResult, PipelineError>>,
}

impl InferencePool {
    /// Spawns a worker that owns `pipeline` and processes parse jobs from a bounded queue.
    pub fn new(pipeline: DocumentPipeline, queue_depth: usize) -> Self {
        let depth = queue_depth.max(1);
        let (tx, rx) = sync_channel::<InferenceJob>(depth);
        let worker = thread::spawn(move || {
            while let Ok(job) = rx.recv() {
                let result = (|| {
                    let img = image::load_from_memory_with_format(&job.bytes, job.format)
                        .map_err(PipelineError::Image)?;
                    pipeline.parse_image(img, job.filename)
                })();
                let _ = job.reply.send(result);
            }
        });
        Self {
            tx,
            _worker: worker,
        }
    }

    pub async fn parse_image(
        &self,
        bytes: Vec<u8>,
        format: ImageFormat,
        filename: Option<String>,
    ) -> Result<DocumentParseResult, PipelineError> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(InferenceJob {
                bytes,
                format,
                filename,
                reply: reply_tx,
            })
            .map_err(|_| PipelineError::Message("inference queue closed".into()))?;
        reply_rx.await.map_err(|_| {
            PipelineError::Message("inference worker dropped response channel".into())
        })?
    }
}
