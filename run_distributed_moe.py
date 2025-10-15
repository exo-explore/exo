#!/usr/bin/env python3
"""
Run distributed MoE test using MLX distributed directly
"""

import subprocess
import sys
import os

# First, let's use the simpler mpirun approach with proper Python paths
def main():
    # Sync files to mini2
    print("Syncing files to mini2...")
    sync_commands = [
        "ssh mini2@192.168.5.2 'mkdir -p /Users/mini2/Movies/exo/exo/inference/mlx/models'",
        "scp -q exo/inference/shard.py mini2@192.168.5.2:/Users/mini2/Movies/exo/exo/inference/",
        "scp -q exo/inference/mlx/models/base.py mini2@192.168.5.2:/Users/mini2/Movies/exo/exo/inference/mlx/models/",
        "scp -q exo/inference/mlx/models/qwen_moe_mini.py mini2@192.168.5.2:/Users/mini2/Movies/exo/exo/inference/mlx/models/",
        "scp -q test_moe_distributed.py mini2@192.168.5.2:/Users/mini2/Movies/exo/",
    ]
    
    for cmd in sync_commands:
        subprocess.run(cmd, shell=True, capture_output=True)
    
    print("Files synced!")
    
    # Get current Python interpreter
    python_exe = sys.executable
    print(f"Using Python: {python_exe}")
    
    # Copy wrapper script to mini2
    subprocess.run("scp -q run_moe_mini2.sh mini2@192.168.5.2:/Users/mini2/Movies/exo/", shell=True)
    subprocess.run("ssh mini2@192.168.5.2 'chmod +x /Users/mini2/Movies/exo/run_moe_mini2.sh'", shell=True)
    
    # Run with mpirun using the wrapper script on mini2
    print("\nLaunching distributed test...")
    
    cmd = f"""mpirun \
        -n 1 -host localhost {python_exe} test_moe_distributed.py --mock-weights : \
        -n 1 -host 192.168.5.2 /Users/mini2/Movies/exo/run_moe_mini2.sh \
        --mca btl_tcp_if_include 192.168.5.0/24"""
    
    # Execute command
    print(f"Running: {cmd}")
    
    result = subprocess.run(cmd, shell=True)
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())