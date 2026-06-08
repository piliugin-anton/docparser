//! Lazy-initialized model runner behind a mutex (shared by inference facades).

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

/// Access failure for a [`LazyRunner`] guard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LazyRunnerAccessError {
    LockPoisoned,
    RunnerNotLoaded,
}

/// Defers loading `R` until first use; loads from `model_dir` via the provided closure.
pub struct LazyRunner<R> {
    label: &'static str,
    model_dir: PathBuf,
    runner: Mutex<Option<R>>,
}

impl<R> LazyRunner<R> {
    pub fn new(label: &'static str, model_dir: impl Into<PathBuf>) -> Self {
        Self {
            label,
            model_dir: model_dir.into(),
            runner: Mutex::new(None),
        }
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    pub fn with_runner<E, T>(
        &self,
        load: impl FnOnce(&Path) -> Result<R, E>,
        f: impl FnOnce(&R) -> Result<T, E>,
    ) -> Result<T, E>
    where
        E: From<LazyRunnerAccessError>,
    {
        let guard = self.guard_mut(load)?;
        let runner = guard
            .as_ref()
            .ok_or(LazyRunnerAccessError::RunnerNotLoaded)?;
        f(runner)
    }

    pub fn with_runner_mut<E, T>(
        &self,
        load: impl FnOnce(&Path) -> Result<R, E>,
        f: impl FnOnce(&mut R) -> Result<T, E>,
    ) -> Result<T, E>
    where
        E: From<LazyRunnerAccessError>,
    {
        let mut guard = self.guard_mut(load)?;
        let runner = guard
            .as_mut()
            .ok_or(LazyRunnerAccessError::RunnerNotLoaded)?;
        f(runner)
    }

    fn guard_mut<E>(
        &self,
        load: impl FnOnce(&Path) -> Result<R, E>,
    ) -> Result<std::sync::MutexGuard<'_, Option<R>>, E>
    where
        E: From<LazyRunnerAccessError>,
    {
        let mut guard = self
            .runner
            .lock()
            .map_err(|_| LazyRunnerAccessError::LockPoisoned)?;
        if guard.is_none() {
            let started = Instant::now();
            let runner = load(&self.model_dir)?;
            tracing::info!(
                model = self.label,
                path = %self.model_dir.display(),
                elapsed_ms = started.elapsed().as_millis() as u64,
                "lazy-loaded inference model"
            );
            *guard = Some(runner);
        }
        Ok(guard)
    }
}
