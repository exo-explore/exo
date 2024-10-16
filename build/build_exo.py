import subprocess
import sys
import os 
def run():

    python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"

    command = [
        "python3", "-m", "nuitka", "exo/main.py",
        "--company-name=exolabs",
        "--output-dir=dist",
        "--follow-imports",
        "--standalone",
        "--output-filename=exo"
    ]

    if sys.platform == "darwin":  
        command.extend([
            "--macos-app-name=exo",
            "--macos-app-mode=gui",
            "--macos-app-version=0.0.1",
            "--product-name=exo",
            "--macos-create-app-bundle",
            "--macos-app-icon=docs/exo-logo.icns"
        ])
    elif sys.platform == "win32":  
        command.extend([
            "--windows-icon-from-ico=docs/exo-logo.ico"  
        ])
    elif sys.platform.startswith("linux"):  
        command.extend([
            "--linux-icon=docs/exo-logo.png"
        ])

    command.extend([
        "--include-distribution-meta=pygments",
        "--include-distribution-meta=mlx",
        "--include-module=mlx._reprlib_fix",
        "--include-module=mlx._os_warning",
        f"--include-data-files=./.venv/lib/{python_version}/site-packages/mlx/lib/mlx.metallib=mlx/lib/mlx.metallib",
        "--include-data-dir=exo/tinychat=tinychat",
        "--include-module=exo.inference.mlx.models.llama",
        "--include-module=exo.inference.mlx.models.deepseek_v2",
        "--include-module=exo.inference.mlx.models.base",
        "--include-module=exo.inference.mlx.models.llava",
        "--include-module=exo.inference.mlx.models.qwen2"
    ])

    try:
        subprocess.run(command, check=True)
        print("Build completed!")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    run()