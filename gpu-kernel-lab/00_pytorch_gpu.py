import time
import torch


device = "cuda"

def matmul_time_test(n):
    a = torch.randn(n, n, device=device)
    b = torch.randn(n, n, device=device)

    # warmup
    for _ in range(10):
        c = a @ b

    torch.cuda.synchronize()

    start = time.perf_counter()

    for _ in range(100):
        c = a @ b

    torch.cuda.synchronize()
    end = time.perf_counter()

    print(f"n={n}, average:{(end - start) / 100}")

def main():
    matmul_time_test(1024)
    matmul_time_test(2048)
    matmul_time_test(4096)

if __name__ == "__main__":
    main()

"""
1. warmup 第一次执行可能包括
    - library initialization
    - kernel loading
    - JIT compilation
    - cache effects
上面这些不应该算在 steady-state performance中，所以性能测试通常是 warmup --> benchmark

2. GPU 操作通常是异步的，执行c = a @ b，不代表 GPU 此时已经算完了矩阵乘法，如果不加`torch.cuda.synchronize()`，测到的时间只是 launch kernel 的 CPU 时间，而不是 GPU 真正执行计算的时间.
"""