import os
import importlib.metadata
import importlib.util
import json
import sys


def calc_container(path):
  """Calculate total size of a directory or file."""
  if os.path.isfile(path):
    try:
      return os.path.getsize(path)
    except (OSError, FileNotFoundError):
      return 0

  total_size = 0
  for dirpath, dirnames, filenames in os.walk(path):
    for f in filenames:
      fp = os.path.join(dirpath, f)
      try:
        total_size += os.path.getsize(fp)
      except (OSError, FileNotFoundError):
        continue
  return total_size


def get_package_location(package_name):
  """Get the actual location of a package's files."""
  try:
    spec = importlib.util.find_spec(package_name)
    if spec is None:
      return None

    if spec.submodule_search_locations:
      # Return the first location for namespace packages
      return spec.submodule_search_locations[0]
    elif spec.origin:
      # For single-file modules, return the file path itself
      return spec.origin
  except ImportError:
    return None


def get_package_sizes(min_size_mb=0.1):
  """Get sizes of installed packages above minimum size threshold."""
  package_sizes = []

  # Get all installed distributions
  for dist in importlib.metadata.distributions():
    try:
      package_name = dist.metadata["Name"]
      location = get_package_location(package_name.replace("-", "_"))

      if location and os.path.exists(location):
        size = calc_container(location)
        size_mb = size / (1024 * 1024)

        if size_mb > min_size_mb:
          package_sizes.append((package_name, size))
    except Exception as e:
      print(
        f"Error processing {dist.metadata.get('Name', 'Unknown package')}: {e}"
      )

  return package_sizes


def main():
  # Get and sort package sizes
  package_sizes = get_package_sizes()
  package_sizes.sort(key=lambda x: x[1], reverse=True)

  # Convert sizes to MB and prepare data
  table_data = [(name, size/(1024*1024)) for name, size in package_sizes]
  total_size = sum(size for _, size in package_sizes)/(1024*1024)

  # Check if --json flag is present
  if "--json" in sys.argv:
    try:
      output_file = sys.argv[sys.argv.index("--json") + 1]
      json_data = {
        "packages": [{
          "name": name,
          "size_mb": round(size, 2)
        } for name, size in table_data],
        "total_size_mb": round(total_size, 2)
      }

      with open(output_file, 'w') as f:
        json.dump(json_data, f, indent=2)
      print(f"JSON data written to {output_file}")
      return
    except IndexError:
      print("Error: Please provide a filename after --json")
      sys.exit(1)
    except Exception as e:
      print(f"Error writing JSON file: {e}")
      sys.exit(1)

  # Original table output code
  max_name_width = max(len(name) for name, _ in table_data)
  max_name_width = max(max_name_width, len("Package"))

  print(f"\n{'Package':<{max_name_width}} | Size (MB)")
  print("-" * max_name_width + "-+-" + "-" * 10)

  for name, size in table_data:
    print(f"{name:<{max_name_width}} | {size:>8.2f}")

  print(f"\nTotal size: {total_size:.2f} MB\n")

if __name__ == "__main__":
  main()