import site
import subprocess
import sys
import os 
import pkgutil

def run():
    site_packages = site.getsitepackages()[0]
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    baseimages_dir = os.path.join(base_dir, "exo", "apputil", "baseimages")
    
    command = [
        f"{sys.executable}", "-m", "nuitka", "exo/main.py",
        "--company-name=exolabs",
        "--product-name=exo",
        "--output-dir=dist",
        "--follow-imports",
        "--standalone",
        "--output-filename=exo",
        "--python-flag=no_site",
        "--onefile",
        f"--include-data-dir={baseimages_dir}=exo/apputil/baseimages"
    ]

    if sys.platform == "darwin": 
        command.extend([
            "--macos-app-name=exo",
            "--macos-app-mode=gui",
            "--macos-app-version=0.0.1",
            "--macos-signed-app-name=net.exolabs.exo",
            "--include-distribution-meta=mlx",
            "--include-module=mlx._reprlib_fix",
            "--include-module=mlx._os_warning",
            "--include-distribution-meta=huggingface_hub",
            "--include-module=huggingface_hub.repocard",
            f"--include-data-files={site_packages}/mlx/lib/mlx.metallib=mlx/lib/mlx.metallib",
            f"--include-data-files={site_packages}/mlx/lib/mlx.metallib=./mlx.metallib",
            "--include-distribution-meta=pygments",
            "--nofollow-import-to=tinygrad"
        ])
        inference_modules = [
            name for _, name, _ in pkgutil.iter_modules(['exo/inference/mlx/models'])
        ]
        for module in inference_modules:
            command.append(f"--include-module=exo.inference.mlx.models.{module}")
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
