#!/usr/bin/env python
import subprocess
import sys
import os


def run_yapf(target):
  if os.path.isfile(target):
    files = [target]
  else:
    files = [os.path.join(root, file) for root, _, files in os.walk(target) for file in files if file.endswith('.py')]

  for file in files:
    try:
      command = ["yapf", "-i", file]
      subprocess.run(command, check=True, capture_output=True, text=True)
      print(f"Formatted: {file}")
    except subprocess.CalledProcessError as e:
      print(f"Error formatting {file}: {e.stderr}")


def main():
  if len(sys.argv) < 2:
    print("Usage: python3 format.py <directory_or_file> e.g. python3 format.py ./exo")
    sys.exit(1)

  target = sys.argv[1]
  run_yapf(target)
  print("Formatting completed.")


if __name__ == "__main__":
  main()
