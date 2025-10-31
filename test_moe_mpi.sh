#!/bin/bash

# Simple MPI test for MoE model

echo "Testing MoE with MPI directly..."

# Copy files to mini2
echo "Syncing to mini2..."
ssh mini2@192.168.5.2 "mkdir -p /Users/mini2/Movies/exo/exo/inference/mlx/models" 2>/dev/null
scp -q exo/inference/shard.py mini2@192.168.5.2:/Users/mini2/Movies/exo/exo/inference/
scp -q exo/inference/mlx/models/base.py mini2@192.168.5.2:/Users/mini2/Movies/exo/exo/inference/mlx/models/
scp -q exo/inference/mlx/models/qwen_moe_mini.py mini2@192.168.5.2:/Users/mini2/Movies/exo/exo/inference/mlx/models/
scp -q test_moe_distributed.py mini2@192.168.5.2:/Users/mini2/Movies/exo/

# Create host file for MPI
cat > mpi_hostfile << EOF
localhost slots=1
192.168.5.2 slots=1
EOF

# Run with mpirun
echo "Launching with mpirun..."
mpirun \
    -n 2 \
    --hostfile mpi_hostfile \
    --mca btl_tcp_if_include 192.168.5.0/24 \
    python test_moe_distributed.py --mock-weights