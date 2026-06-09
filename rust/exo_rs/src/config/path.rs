use clap::builder::{PathBufValueParser, TypedValueParser};
use std::path::PathBuf;
use tokio::fs;
use util::path::resolve_path;

/// Default path parser that should be used to ensure paths are
/// resolved to absolute paths before being further processed.
pub fn parse_path() -> impl TypedValueParser<Value = PathBuf> {
    PathBufValueParser::new().try_map(resolve_path)
}

// extension trait to tack on extra validation on path parsing
pub trait PathBufValueParserExt: TypedValueParser<Value = PathBuf> {
    #[inline]
    fn canonicalize(self) -> impl TypedValueParser<Value = PathBuf> {
        self.try_map(|p| p.canonicalize())
    }

    #[inline]
    fn create_dir(self) -> impl TypedValueParser<Value = PathBuf> {
        self.try_map(|p| fs::create_dir_all(p))
    }

    #[inline]
    fn create_parent_dir(self) -> impl TypedValueParser<Value = PathBuf> {
        self.try_map(|p| {
            if let Some(parent) = p.parent() {
                fs::create_dir_all(parent)
            } else {
                Ok(p)
            }
        })
    }
}

impl<T: TypedValueParser<Value = PathBuf>> PathBufValueParserExt for T {}
