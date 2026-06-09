use extend::ext;
use path_clean::PathClean;
use std::path::{Component, Path, PathBuf};
use std::{io, path};

#[ext(pub, name = PathExt)]
impl Path {
    /// Converts path to UTF-8 string, or returns `Err` if not UTF-8.
    #[inline]
    fn to_str_utf8(&self) -> io::Result<&str> {
        self.to_str().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidFilename,
                "path contains non-utf8 characters",
            )
        })
    }
}

/// Resolves any path to its true absolute form as much as possible.
///
/// The path does not need to exist, but some component of it may exist.
/// Expands any `~` or `~user`; eliminates any `..` or `.` and resolves
/// symlinks by traversing the file system for the real part of the path,
/// and performing lexical cleaning for the nonexistent part.
#[cfg(unix)]
pub fn resolve_path(path: PathBuf) -> io::Result<PathBuf> {
    // expand user if possible
    let mut path = expanduser::expanduser(path.to_str_utf8()?)?;

    // convert to absolute, this will slightly clean path even if NOT relative
    path = path::absolute(&path)?;

    let mut components = path.components().collect::<Vec<_>>();
    components
        .first()
        .filter(|&&c| c == Component::RootDir)
        .expect("the first component must exist, and be the root directory");

    // resolve real prefix of path with `Path::canonicalize` (which will follow symlinks)
    // and nonexistent suffix with `PathClean::clean` (which will do lexical cleaning).
    //
    // 1) canonicalization is attempted iteratively to determine prefix/suffix split
    // 2) lexical cleaning ran on suffix which result in leading ".." components
    fn split_canonicalize(components: &[Component]) -> io::Result<(PathBuf, PathBuf)> {
        let mut prefix = PathBuf::new();
        let mut suffix = PathBuf::new();
        for i in (1..=components.len()).rev() {
            prefix = PathBuf::from_iter(&components[..i]);
            suffix = PathBuf::from_iter(&components[i..]);

            match prefix.canonicalize() {
                Ok(p) => {
                    prefix = p;

                    // clean + substitute "." with empty buffer
                    suffix = suffix.clean();
                    if suffix == Path::new(".") {
                        suffix = PathBuf::new()
                    }
                    break;
                }
                Err(e) if i > 1 && e.kind() == io::ErrorKind::NotFound => continue,
                Err(e) => return Err(e),
            }
        }
        Ok((prefix, suffix))
    }
    let (mut prefix, suffix) = split_canonicalize(&components)?;
    // 3) the ".." components are joined with real prefix and canonicalized again
    let mut suffix_components = suffix.components().collect::<Vec<_>>();
    suffix_components.reverse();
    while let Some(&c) = suffix_components.last() {
        if c != Component::ParentDir {
            break;
        }
        prefix.push(
            suffix_components
                .pop()
                .expect("already checked that its non-empty"),
        );
    }
    suffix_components.reverse();
    prefix = prefix.canonicalize()?;

    // 4) prefix/suffix joined and 1) & 2) ran again to resolve any new exposed symlinks
    //    NOTE: this time there shouldn't be any ".." in suffix
    prefix.extend(suffix_components);
    components = prefix.components().collect::<Vec<_>>();
    let (mut prefix, suffix) = split_canonicalize(&components)?;
    assert!(
        suffix
            .components()
            .all(|c| !matches!(c, Component::ParentDir | Component::CurDir)),
        "all `.` or `..` in suffix should have been eliminated"
    );

    // 5) prefix/suffix joined in final absolute clean path with symlinks resolved and no ".."
    prefix.push(&suffix);
    let path = prefix;
    assert!(
        path.components()
            .all(|c| !matches!(c, Component::ParentDir | Component::CurDir)),
        "all `.` or `..` in path should have been eliminated"
    );

    Ok(path)
}
