#!/bin/bash

# Ensure both systems have identical setup for MoE testing

set -e

echo "=== Synchronizing MoE test environment ==="

# 1. Create identical directory structure on mini2
echo "Creating directories on mini2..."
ssh mini2@192.168.5.2 "mkdir -p /tmp/mlx_moe_test"

# 2. Copy all necessary files to /tmp on both machines (same path)
echo "Setting up local /tmp/mlx_moe_test..."
mkdir -p /tmp/mlx_moe_test
cp -r exo /tmp/mlx_moe_test/
cp test_moe_distributed.py /tmp/mlx_moe_test/
cp test_moe_ring.py /tmp/mlx_moe_test/

# 3. Sync to mini2
echo "Syncing to mini2..."
rsync -avq /tmp/mlx_moe_test/ mini2@192.168.5.2:/tmp/mlx_moe_test/

# 4. Create identical Python wrapper on both machines
cat > /tmp/mlx_moe_test/run_test.py << 'EOF'
#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, '/tmp/mlx_moe_test')
os.chdir('/tmp/mlx_moe_test')

# Import and run the test
import test_moe_ring
test_moe_ring.main()
EOF

# Copy wrapper to mini2
scp -q /tmp/mlx_moe_test/run_test.py mini2@192.168.5.2:/tmp/mlx_moe_test/

# 5. Test with mpirun using identical paths
echo "Testing distributed MoE..."
cd /tmp/mlx_moe_test

# Get Python path
PYTHON_PATH=$(which python3)
echo "Using Python: $PYTHON_PATH"

# Run with mpirun
echo "Launching distributed test..."
mpirun \
    -n 1 -host localhost $PYTHON_PATH /tmp/mlx_moe_test/run_test.py : \
    -n 1 -host 192.168.5.2 python3 /tmp/mlx_moe_test/run_test.py \
    --mca btl_tcp_if_include 192.168.5.0/24

echo "=== Test complete ==="