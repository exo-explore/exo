import subprocess
import sys
import os 
def run():

    python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"

    command = [
        f"{sys.executable}", "-m", "nuitka", "exo/main.py",
        "--company-name=exolabs",
        "--product-name=exo",
        "--output-dir=dist",
        "--follow-imports",
        "--standalone",
        "--output-filename=exo",
        "--static-libpython=yes"
    ]

    if sys.platform == "darwin": 
        command.extend([
            "--macos-app-name=exo",
            "--macos-app-mode=background",
            "--macos-app-version=0.0.1",
            "--onefile",
            "--macos-create-app-bundle",
            "--macos-app-icon=docs/exo-logo.icns",
            "--include-module=exo.inference.mlx.models.llama",
            "--include-module=exo.inference.mlx.models.deepseek_v2",
            "--include-module=exo.inference.mlx.models.base",
            "--include-module=exo.inference.mlx.models.llava",
            "--include-module=exo.inference.mlx.models.qwen2",
            "--include-distribution-meta=mlx",
            "--include-module=mlx._reprlib_fix",
            "--include-module=mlx._os_warning",
            f"--include-data-files=./.venv/lib/{python_version}/site-packages/mlx/lib/mlx.metallib=mlx/lib/mlx.metallib",
            f"--include-data-files=./.venv/lib/{python_version}/site-packages/mlx/lib/mlx.metallib=./mlx.metallib",
            "--include-distribution-meta=pygments"
        ])
    elif sys.platform == "win32":  
        command.extend([
            "--onefile",
            "--windows-icon-from-ico=docs/exo-logo-win.ico",
            "--file-version=0.0.1",
            "--product-version=0.0.1"
        ])
    elif sys.platform.startswith("linux"):  
        command.extend([
            "--include-distribution-metadata=pygments",
            "--linux-icon=docs/exo-rounded.png"
        ])

    command.extend([
        "--include-data-dir=exo/tinychat=tinychat"
    ])

    if sys.platform == 'darwin':
        libpython_path = f"/opt/homebrew/opt/python@{sys.version_info.major}.{sys.version_info.minor}/Frameworks/Python.framework/Versions/{sys.version_info.major}.{sys.version_info.minor}/lib/python{sys.version_info.major}.{sys.version_info.minor}/config-{sys.version_info.major}.{sys.version_info.minor}-darwin"
        include_path = f"/opt/homebrew/opt/python@{sys.version_info.major}.{sys.version_info.minor}/Frameworks/Python.framework/Versions/{sys.version_info.major}.{sys.version_info.minor}/include/python{sys.version_info.major}.{sys.version_info.minor}"
        env = os.environ.copy()
        env['LDFLAGS'] = f"-L{libpython_path}"
        env['CPPFLAGS'] = f"-I{include_path}"

    try:
        subprocess.run(command, check=True)
        print("Build completed!")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    run()