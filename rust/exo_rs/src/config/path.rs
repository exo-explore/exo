use clap::builder::{PathBufValueParser, TypedValueParser};
use std::path::PathBuf;
use util::path::resolve_path;

pub fn parse_path() -> impl TypedValueParser<Value = PathBuf> {
    PathBufValueParser::new().try_map(resolve_path)
}
