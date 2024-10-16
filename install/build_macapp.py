import subprocess

def build_macapp():
    command = [
        "python3", "-m", "nuitka", "exo/main.py",
        "--output-dir=dist",
        "--macos-app-name=exo",
        "--macos-app-mode=gui",
        "--macos-app-version=0.0.1",
        "--product-name=exo",
        "--follow-imports",
        "--standalone",
        "--macos-create-app-bundle",
        "--macos-app-icon=/Users/eyasunigussie/exo/docs/exo-logo.icns",
        "--include-distribution-meta=pygments",
        "--include-distribution-meta=mlx",
        "--include-module=mlx._reprlib_fix",
        "--include-module=mlx._os_warning",
        "--output-filename=exo",
        "--include-data-files=./.venv/lib/python3.11/site-packages/mlx/lib/mlx.metallib=mlx/lib/mlx.metallib",
        "--include-data-dir=exo/tinychat=tinychat",
        "--include-package=models",
        "--include-module=exo.inference.mlx.models.llama",
        "--include-module=exo.inference.mlx.models.deepseek_v2",
        "--include-module=exo.inference.mlx.models.base",
        "--include-module=exo.inference.mlx.models.llava",
        "--include-module=exo.inference.mlx.models.qwen2"
    ]
    
    try:
        subprocess.run(command, check=True)
        print("Build completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
    
if __name__ == "__main__":
    build_macapp()