import site
import subprocess
import sys
import os 
def run():
    site_packages = site.getsitepackages()[0]
    command = [
        f"{sys.executable}", "-m", "nuitka", "exo/main.py",
        "--company-name=exolabs",
        "--product-name=exo",
        "--output-dir=dist",
        "--follow-imports",
        "--onefile",
        "--standalone",
        "--output-filename=exo"
    ]

    if sys.platform == "darwin": 
        command.extend([
            "--macos-app-name=exo",
            "--macos-app-mode=background",
            "--macos-app-version=0.0.1",
            "--include-module=exo.inference.mlx.models.llama",
            "--include-module=exo.inference.mlx.models.deepseek_v2",
            "--include-module=exo.inference.mlx.models.base",
            "--include-module=exo.inference.mlx.models.llava",
            "--include-module=exo.inference.mlx.models.qwen2",
            "--include-distribution-meta=mlx",
            "--include-module=mlx._reprlib_fix",
            "--include-module=mlx._os_warning",
            f"--include-data-files={site_packages}/mlx/lib/mlx.metallib=mlx/lib/mlx.metallib",
            f"--include-data-files={site_packages}/mlx/lib/mlx.metallib=./mlx.metallib",
            "--include-distribution-meta=pygments"
        ])
    elif sys.platform == "win32":  
        command.extend([
            "--windows-icon-from-ico=docs/exo-logo-win.ico",
            "--file-version=0.0.1",
            "--product-version=0.0.1"
        ])
    elif sys.platform.startswith("linux"):  
        command.extend([
            "--include-distribution-metadata=pygments",
            "--linux-icon=docs/exo-rounded.png"
        ])


    try:
        subprocess.run(command, check=True)
        print("Build completed!")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    run()