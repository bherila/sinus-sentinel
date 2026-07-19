//! Per-user single-instance ownership and activation handoff.
//!
//! The operating system owns the file lock, so it is released automatically on
//! crashes and cannot leave a stale PID lock behind. A contending process writes
//! a marker in the same private app-data directory and exits before opening the
//! database or starting audio/sync workers. The owner consumes that marker from
//! its UI tick and reveals its History window.

use std::fs::{File, OpenOptions, TryLockError};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

const LOCK_FILE: &str = "instance.lock";
const ACTIVATE_FILE: &str = "activate.request";

pub enum AcquireOutcome {
    Primary(InstanceGuard),
    ActivatedExisting,
}

pub struct InstanceGuard {
    _lock_file: File,
    activation_path: PathBuf,
}

impl InstanceGuard {
    pub fn acquire(data_dir: &Path) -> io::Result<AcquireOutcome> {
        let activation_path = data_dir.join(ACTIVATE_FILE);
        // Discard a marker left by a process that died before consuming it. Every
        // contender writes a fresh marker after it observes the held lock.
        match std::fs::remove_file(&activation_path) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }

        let lock_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(data_dir.join(LOCK_FILE))?;

        match lock_file.try_lock() {
            Ok(()) => Ok(AcquireOutcome::Primary(InstanceGuard {
                _lock_file: lock_file,
                activation_path,
            })),
            Err(TryLockError::WouldBlock) => {
                signal_activation(&activation_path)?;
                Ok(AcquireOutcome::ActivatedExisting)
            }
            Err(TryLockError::Error(error)) => Err(error),
        }
    }

    /// Consume one or more coalesced activation requests.
    pub fn take_activation_request(&self) -> bool {
        match std::fs::remove_file(&self.activation_path) {
            Ok(()) => true,
            Err(error) if error.kind() == io::ErrorKind::NotFound => false,
            Err(error) => {
                eprintln!("single-instance: could not consume activation request: {error}");
                false
            }
        }
    }
}

fn signal_activation(path: &Path) -> io::Result<()> {
    let mut request = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;
    request.write_all(b"show-history\n")?;
    request.sync_all()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_dir() -> PathBuf {
        std::env::temp_dir().join(format!("sinus-instance-{}", uuid::Uuid::new_v4()))
    }

    #[test]
    fn contender_activates_owner_and_crash_release_needs_no_stale_cleanup() {
        let dir = test_dir();
        std::fs::create_dir_all(&dir).unwrap();

        let owner = match InstanceGuard::acquire(&dir).unwrap() {
            AcquireOutcome::Primary(owner) => owner,
            AcquireOutcome::ActivatedExisting => panic!("first process must own the lock"),
        };
        assert!(matches!(
            InstanceGuard::acquire(&dir).unwrap(),
            AcquireOutcome::ActivatedExisting
        ));
        assert!(owner.take_activation_request());
        assert!(!owner.take_activation_request());

        // Dropping the file simulates process termination; the next launch owns
        // the operating-system lock without PID checks or stale-lock deletion.
        drop(owner);
        assert!(matches!(
            InstanceGuard::acquire(&dir).unwrap(),
            AcquireOutcome::Primary(_)
        ));

        std::fs::remove_dir_all(dir).ok();
    }
}
