use clap::builder::{PathBufValueParser, TypedValueParser};
use std::path::PathBuf;
use std::{fs, io};
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
    fn dir_exists(self) -> impl TypedValueParser<Value = PathBuf> {
        self.canonicalize().try_map(|p| {
            let m = fs::metadata(&p)?;
            if m.is_dir() {
                Ok(p)
            } else {
                Err(io::Error::new(
                    io::ErrorKind::NotADirectory,
                    format!("{p:?} is not a directory"),
                ))
            }
        })
    }

    #[inline]
    fn file_exists(self) -> impl TypedValueParser<Value = PathBuf> {
        self.canonicalize().try_map(|p| {
            let m = fs::metadata(&p)?;
            if !m.is_dir() {
                Ok(p)
            } else {
                Err(io::Error::new(
                    io::ErrorKind::IsADirectory,
                    format!("{p:?} is a directory"),
                ))
            }
        })
    }

    #[inline]
    fn create_dir(self) -> impl TypedValueParser<Value = PathBuf> {
        self.try_map(|p| -> io::Result<_> {
            fs::create_dir_all(&p)?;
            Ok(p)
        })
    }

    #[inline]
    fn create_parent_dir(self) -> impl TypedValueParser<Value = PathBuf> {
        self.try_map(|p| -> io::Result<_> {
            if let Some(parent) = p.parent() {
                fs::create_dir_all(parent)?;
            }
            Ok(p)
        })
    }
}

impl<T: TypedValueParser<Value = PathBuf>> PathBufValueParserExt for T {}
