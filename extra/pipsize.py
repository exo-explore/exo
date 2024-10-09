import os
import pkg_resources
from tabulate import tabulate

def calc_container(path):
  total_size = 0
  for dirpath, dirnames, filenames in os.walk(path):
    for f in filenames:
      fp = os.path.join(dirpath, f)
      total_size += os.path.getsize(fp)
  return total_size

dists = [d for d in pkg_resources.working_set]
package_sizes = []

for dist in dists:
  try:
    path = os.path.join(dist.location, dist.project_name)
    size = calc_container(path)
    if size / (1024 * 1024) > 0.1:  # Only include packages larger than 0.1 MB
      package_sizes.append((dist.project_name, size))
  except OSError:
    print(f'{dist.project_name} no longer exists')

# Sort packages by size in descending order
package_sizes.sort(key=lambda x: x[1], reverse=True)

# Convert sizes to MB and prepare data for tabulation
table_data = [(name, f"{size/(1024*1024):.2f}") for name, size in package_sizes]

# Print sorted packages in a tabular format
headers = ["Package", "Size (MB)"]
print(tabulate(table_data, headers=headers, tablefmt="grid"))
