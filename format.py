#!/usr/bin/env python
import re
import subprocess
import sys
import os
import fnmatch

DEBUG_PATTERN = re.compile(r'^(\s*)(if\s+DEBUG\s*>=?\s*\d+\s*:.+)$', re.MULTILINE)
PLACEHOLDER = "###DEBUG_PLACEHOLDER###"

# Add ignore patterns here
IGNORE_PATTERNS = [
    '.venv/*',
    'setup.py',
    '*helpers.py',
    '*node_service_pb2.py',
    '*node_service_pb2_grpc.py',
]

def should_ignore(file_path):
    for pattern in IGNORE_PATTERNS:
        if fnmatch.fnmatch(file_path, pattern):
            return True
    return False

def preserve_debug_lines(content):
    def replace(match):
        indent, line = match.groups()
        return f"{indent}{PLACEHOLDER}{line.strip()}"
    return DEBUG_PATTERN.sub(replace, content)

def restore_debug_lines(content):
    return re.sub(f"^(\\s*){PLACEHOLDER}(.+)$", r"\1\2", content, flags=re.MULTILINE)

def adjust_indentation(content):
    lines = content.split('\n')
    adjusted_lines = []
    for line in lines:
        if line.strip() and not line.startswith(PLACEHOLDER):
            indent = len(line) - len(line.lstrip())
            new_indent = ' ' * (indent // 2)
            adjusted_lines.append(new_indent + line.lstrip())
        else:
            adjusted_lines.append(line)
    return '\n'.join(adjusted_lines)

def process_file(file_path, process_func):
    with open(file_path, 'r') as file:
        content = file.read()

    modified_content = process_func(content)

    if content != modified_content:
        with open(file_path, 'w') as file:
            file.write(modified_content)

def run_black(target):
    # Convert ignore patterns to Black's --extend-exclude format
    exclude_patterns = '|'.join(f'({pattern.replace("*", ".*")})' for pattern in IGNORE_PATTERNS)
    command = [
        "black",
        "--line-length", "120",
        "--extend-exclude", exclude_patterns,
        target
    ]
    subprocess.run(command, check=True)

def format_files(target):
    if os.path.isfile(target):
        files = [target] if not should_ignore(target) else []
    elif os.path.isdir(target):
        files = []
        for root, _, filenames in os.walk(target):
            for filename in filenames:
                if filename.endswith('.py'):
                    file_path = os.path.join(root, filename)
                    if not should_ignore(file_path):
                        files.append(file_path)
    else:
        print(f"Error: {target} is not a valid file or directory")
        return

    # Preserve debug lines
    for file in files:
        process_file(file, preserve_debug_lines)

    # Run Black
    run_black(target)

    # Adjust indentation and restore debug lines
    for file in files:
        process_file(file, adjust_indentation)
        process_file(file, restore_debug_lines)

def main():
    if len(sys.argv) < 2:
        print("Usage: python format.py <directory_or_file>")
        sys.exit(1)

    target = sys.argv[1]
    format_files(target)
    print("Formatting completed.")

if __name__ == "__main__":
    main()