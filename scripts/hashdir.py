import hashlib
import os
import sys

EXCLUDE_DIRS = {".git", "build", "vendor", ".idea", ".vscode", "__pycache__"}

def norm_rel(path: str, base: str) -> str:
    """Forwarder-root–relative path with '/' separators."""
    abs_path = os.path.abspath(path)
    abs_base = os.path.abspath(base)
    rel = os.path.relpath(abs_path, abs_base)
    return rel.replace(os.sep, "/")

def collect_files(arg_path: str) -> tuple[str, list[str]]:
    # Resolve forwarder_root and src_root from the provided path
    p = os.path.abspath(arg_path)
    if not os.path.isdir(p):
        sys.stderr.write(f"error: path must be a directory: {arg_path}\n")
        sys.exit(2)

    if os.path.basename(p) == "src":
        forwarder_root = os.path.dirname(p)
        src_root = p
    else:
        forwarder_root = p
        src_root = os.path.join(forwarder_root, "src")

    files = []

    # 1) Include .go files under src, excluding *_test.go
    if os.path.isdir(src_root):
        for root, dirs, filenames in os.walk(src_root):
            # prune excluded dirs
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            for name in filenames:
                # strict .go, exclude *_test.go
                if not name.lower().endswith(".go"):
                    continue
                if name.lower().endswith("_test.go"):
                    continue
                files.append(os.path.join(root, name))

    # 2) Add go.mod, go.sum, main.go from the forwarder root
    for name in ("go.mod", "go.sum", "main.go"):
        pth = os.path.join(forwarder_root, name)
        if os.path.isfile(pth):
            # defensive: exclude *_test.go at root too
            if name.lower().endswith("_test.go"):
                continue
            files.append(pth)

    # Deduplicate and sort deterministically by forwarder-root–relative path
    files: list[str] = sorted(set(files), key=lambda f: norm_rel(f, forwarder_root))
    return forwarder_root, files

def hash_files(forwarder_root: str, files: list[str]) -> str:
    h = hashlib.sha256()
    for fp in files:
        rel = norm_rel(fp, forwarder_root)
        h.update(b"F\x00")
        h.update(rel.encode("utf-8"))
        h.update(b"\x00")
        with open(fp, "rb") as f:
            for chunk in iter(lambda: f.read(256 * 1024), b""):
                h.update(chunk)
        h.update(b"\n")
    return h.hexdigest()

def main():
    if len(sys.argv) > 1:
        arg = sys.argv[1]
    else:
        arg = os.path.join("networking", "forwarder", "src")
    forwarder_root, files = collect_files(arg)
    digest = hash_files(forwarder_root, files)
    # print without trailing newline (easier to capture in shell)
    sys.stdout.write(digest)

if __name__ == "__main__":
    main()
