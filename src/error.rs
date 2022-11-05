use std::error::Error;
use std::fmt::{self, Debug, Display, Formatter};

#[derive(Debug)]
pub(crate) enum ArtError {
    VersionNotMatch(usize),
    Locked(usize),
    NeedRestart,
    Oom,
    NodeMoved,
}

/// Out of memory error
pub struct OOMError {}

impl OOMError {
    pub(crate) fn new() -> Self {
        Self {}
    }
}

impl Debug for OOMError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Allocator is out of memory!").finish()
    }
}

impl Display for OOMError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "Allocator is out of memory!")
    }
}

impl Error for OOMError {}
