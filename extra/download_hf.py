import argparse
import asyncio
from exo.download.hf.hf_helpers import download_all_files, RepoProgressEvent

DEFAULT_ALLOW_PATTERNS = [
    "*.json",
    "*.py",
    "tokenizer.model",
    "*.tiktoken",
    "*.txt",
    "*.safetensors",
]
# Always ignore `.git` and `.cache/huggingface` folders in commits
DEFAULT_IGNORE_PATTERNS = [
    ".git",
    ".git/*",
    "*/.git",
    "**/.git/**",
    ".cache/huggingface",
    ".cache/huggingface/*",
    "*/.cache/huggingface",
    "**/.cache/huggingface/**",
]

async def main(repo_id, revision="main", allow_patterns=None, ignore_patterns=None):
    async def progress_callback(event: RepoProgressEvent):
        print(f"Overall Progress: {event.completed_files}/{event.total_files} files, {event.downloaded_bytes}/{event.total_bytes} bytes")
        print(f"Estimated time remaining: {event.overall_eta}")
        print("File Progress:")
        for file_path, progress in event.file_progress.items():
            status_icon = {
                'not_started': '⚪',
                'in_progress': '🔵',
                'complete': '✅'
            }[progress.status]
            eta_str = str(progress.eta)
            print(f"{status_icon} {file_path}: {progress.downloaded}/{progress.total} bytes, "
                  f"Speed: {progress.speed:.2f} B/s, ETA: {eta_str}")
        print("\n")

    await download_all_files(repo_id, revision, progress_callback, allow_patterns, ignore_patterns)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download files from a Hugging Face model repository.")
    parser.add_argument("--repo-id", required=True, help="The repository ID (e.g., 'meta-llama/Meta-Llama-3.1-8B-Instruct')")
    parser.add_argument("--revision", default="main", help="The revision to download (branch, tag, or commit hash)")
    parser.add_argument("--allow-patterns", nargs="*", default=None, help="Patterns of files to allow (e.g., '*.json' '*.safetensors')")
    parser.add_argument("--ignore-patterns", nargs="*", default=None, help="Patterns of files to ignore (e.g., '.*')")

    args = parser.parse_args()

    asyncio.run(main(args.repo_id, args.revision, args.allow_patterns, args.ignore_patterns))