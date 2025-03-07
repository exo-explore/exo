import os
from tinygrad.tensor import Tensor
from tinygrad.runtime.ops_gpu import cl as CL

# 设置环境变量以使用 GPU
os.environ["GPU"] = "1"

# 检查是否检测到 GPU
if CL is None:
    print("未检测到 GPU。请确保已正确安装 OpenCL 驱动程序。")
else:
    print("检测到 GPU，开始运行测试...")

    # 创建两个随机张量
    a = Tensor.randn(1000, 1000)
    b = Tensor.randn(1000, 1000)

    # 在 GPU 上执行矩阵乘法
    c = a @ b

    # 将结果移动到 CPU 并计算其和
    result = c.numpy().sum()

    print(f"测试成功，结果为：{result}")

