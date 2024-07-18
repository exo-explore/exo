#!/bin/bash
export LOGOPS=/tmp/ops
rm $LOGOPS

# generate many kernels
PYTHONPATH="." OPT=2 GPU=1 python3 test/external/external_test_opt.py
PYTHONPATH="." OPT=3 GPU=1 python3 test/external/external_test_opt.py
GPU=1 IMAGE=1 python3 test/test_ops.py
FORWARD_ONLY=1 GPU=1 IMAGE=2 python test/test_ops.py
STEPS=3 python3 examples/hlb_cifar10.py
WINO=1 STEPS=3 python3 examples/hlb_cifar10.py
python3 examples/stable_diffusion.py --noshow
python3 examples/llama.py --prompt "hello" --count 5
python3 examples/gpt2.py --count 5
HALF=1 python3 examples/gpt2.py --count 5
python3 examples/beautiful_mnist.py
python3 examples/beautiful_cartpole.py
python3 examples/mlperf/model_spec.py
python3 examples/yolov8.py ./test/models/efficientnet/Chicken.jpg
examples/openpilot/go.sh
JIT=2 BIG=1 MPS=1 pytest test/ --ignore=test/test_fusion_op.py --ignore=test/test_linearizer_failures.py --ignore=test/test_gc.py --ignore=test/test_speed_v_torch.py --ignore=test/test_jit.py
JIT=2 BIG=1 MPS=1 python -m pytest test/test_gc.py
JIT=2 BIG=1 MPS=1 python -m pytest test/test_jit.py
JIT=2 BIG=1 MPS=1 python -m pytest test/test_speed_v_torch.py

# sort and uniq
sort -u /tmp/ops > /tmp/sops
ls -lh /tmp/ops /tmp/sops
