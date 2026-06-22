use extend::ext;
use path_clean::PathClean;
use std::fs::File;
use std::path::{Component, Path, PathBuf};
use std::{fs, io, path};

#[ext(pub, name = PathExt)]
impl Path {
    /// Converts path to UTF-8 string, or returns `Err` if not UTF-8.
    #[inline(always)]
    fn to_str_utf8(&self) -> io::Result<&str> {
        self.to_str().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidFilename,
                "path contains non-utf8 characters",
            )
        })
    }

    #[inline(always)]
    fn create_file_if_not_found(&self) -> io::Result<()> {
        match File::create_new(self) {
            Ok(_) => Ok(()),
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {
                if self.is_dir() {
                    return Err(io::Error::new(
                        io::ErrorKind::IsADirectory,
                        format!("{self:?} is a directory, not a file"),
                    ));
                }
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    #[cfg(unix)]
    #[inline(always)]
    fn try_dir_exists(&self) -> io::Result<()> {
        let m = fs::metadata(self)?;
        if m.is_dir() {
            Ok(())
        } else {
            Err(io::Error::new(
                io::ErrorKind::NotADirectory,
                format!("{self:?} is not a directory"),
            ))
        }
    }

    #[cfg(unix)]
    #[inline(always)]
    fn try_file_exists(&self) -> io::Result<()> {
        let m = fs::metadata(self)?;
        if !m.is_dir() {
            Ok(())
        } else {
            Err(io::Error::new(
                io::ErrorKind::IsADirectory,
                format!("{self:?} is a directory"),
            ))
        }
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
                    // ensure non-leaf components are directories
                    if i != components.len() && !fs::metadata(&p)?.is_dir() {
                        return Err(io::Error::new(
                            io::ErrorKind::AddrInUse,
                            format!(
                                "cannot resolve {:?}: {:?} is not a directory",
                                PathBuf::from_iter(components),
                                prefix
                            ),
                        ));
                    }
                    prefix = p;

                    // clean + substitute "." with empty buffer
                    suffix = suffix.clean();
                    if suffix == Path::new(".") {
                        suffix = PathBuf::new()
                    }
                    break;
                }
                Err(e)
                    if i > 1
                        && matches!(
                            e.kind(),
                            io::ErrorKind::NotFound | io::ErrorKind::NotADirectory
                        ) =>
                {
                    continue;
                }
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

#[cfg(all(test, unix))]
mod tests {
    use super::resolve_path;
    use std::fs;
    use std::os::unix::fs::symlink;
    use std::path::{Component, Path};
    use tempfile::TempDir;

    fn assert_is_root_followed_by_normal_components(path: &Path) {
        let mut components = path.components();

        assert_eq!(
            components.next(),
            Some(Component::RootDir),
            "resolved path should start with root: {}",
            path.display()
        );
        assert!(
            components.all(|component| matches!(component, Component::Normal(_))),
            "resolved path should contain only normal components after root: {}",
            path.display()
        );
    }

    #[test]
    fn resolve_path_lexically_cleans_nonexistent_suffix() {
        let test_dir = TempDir::new().unwrap();
        let base = test_dir.path().join("base");
        fs::create_dir_all(&base).unwrap();

        let resolved = resolve_path(base.join("missing").join("..").join("leaf")).unwrap();

        assert_eq!(resolved, base.canonicalize().unwrap().join("leaf"));
        assert_is_root_followed_by_normal_components(&resolved);
    }

    #[test]
    fn resolve_path_applies_leading_suffix_parents_to_canonical_prefix() {
        let test_dir = TempDir::new().unwrap();
        let base = test_dir.path().join("base");
        fs::create_dir_all(&base).unwrap();

        let resolved = resolve_path(
            base.join("missing")
                .join("..")
                .join("..")
                .join("outside")
                .join("leaf"),
        )
        .unwrap();

        assert_eq!(
            resolved,
            test_dir.path().canonicalize().unwrap().join("outside/leaf")
        );
        assert_is_root_followed_by_normal_components(&resolved);
    }

    #[test]
    fn resolve_path_resolves_symlinks_exposed_by_cleaned_suffix() {
        let test_dir = TempDir::new().unwrap();
        let base = test_dir.path().join("base");
        let real_target = test_dir.path().join("real-target");
        let link = base.join("link");
        fs::create_dir_all(&base).unwrap();
        fs::create_dir_all(&real_target).unwrap();
        symlink(&real_target, &link).unwrap();

        let resolved = resolve_path(
            base.join("missing")
                .join("..")
                .join("link")
                .join("future")
                .join("..")
                .join("leaf"),
        )
        .unwrap();

        assert_eq!(resolved, real_target.canonicalize().unwrap().join("leaf"));
        assert_is_root_followed_by_normal_components(&resolved);
    }

    #[test]
    fn resolve_path_preserves_existing_symlink_parent_semantics() {
        let test_dir = TempDir::new().unwrap();
        let real_parent = test_dir.path().join("real-parent");
        let real_target = real_parent.join("target");
        let link = test_dir.path().join("link");
        fs::create_dir_all(&real_target).unwrap();
        symlink(&real_target, &link).unwrap();

        let resolved = resolve_path(link.join("..")).unwrap();

        assert_eq!(resolved, real_parent.canonicalize().unwrap());
        assert_is_root_followed_by_normal_components(&resolved);
    }
}
