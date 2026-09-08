import torch

from mha_mqa_lab.mha import mha_attention
from mha_mqa_lab.mqa import mqa_attention


def main() -> None:
    torch.manual_seed(42)
    batch, length, heads, head_dim = 1, 4, 4, 4
    model_dim = heads * head_dim
    hidden = torch.randn(batch, length, model_dim)

    wq = torch.randn(model_dim, heads, head_dim)
    mha_wk = torch.randn(model_dim, heads, head_dim)
    mha_wv = torch.randn(model_dim, heads, head_dim)
    mqa_wk = torch.randn(model_dim, head_dim)
    mqa_wv = torch.randn(model_dim, head_dim)

    query = torch.einsum("btd,dhk->bhtk", hidden, wq)
    mha_key = torch.einsum("btd,dhk->bhtk", hidden, mha_wk)
    mha_value = torch.einsum("btd,dhv->bhtv", hidden, mha_wv)
    mqa_key = torch.einsum("btd,dk->btk", hidden, mqa_wk).unsqueeze(1)
    mqa_value = torch.einsum("btd,dv->btv", hidden, mqa_wv).unsqueeze(1)
    mha_output, mha_weights = mha_attention(query, mha_key, mha_value)
    mqa_output, mqa_weights = mqa_attention(query, mqa_key, mqa_value)

    print("=== Dimension ===")
    for name, tensor in (
        ("X", hidden),
        ("Q", query),
        ("MHA K/V", mha_key),
        ("MQA K/V", mqa_key),
        ("MHA scores", mha_weights),
        ("MQA scores", mqa_weights),
        ("MHA output", mha_output),
        ("MQA output", mqa_output),
    ):
        print(f"{name:<18} {list(tensor.shape)}")

    shared_key = torch.randn(batch, 1, length, head_dim)
    shared_value = torch.randn_like(shared_key)
    tied_output, _ = mha_attention(
        query,
        shared_key.expand(-1, heads, -1, -1),
        shared_value.expand(-1, heads, -1, -1),
    )
    mqa_output, _ = mqa_attention(query, shared_key, shared_value)
    torch.testing.assert_close(tied_output, mqa_output)

    layers, context, model_heads, model_head_dim = 32, 4096, 32, 128
    model_dim = model_heads * model_head_dim
    mha_bytes = 2 * layers * context * model_heads * model_head_dim * 2
    mqa_bytes = 2 * layers * context * model_head_dim * 2
    flops = 4 * layers * context * model_heads * model_head_dim
    mha_parameters = 4 * model_dim**2
    mqa_parameters = 2 * model_dim**2 + 2 * model_dim * model_head_dim
    print("\n=== 32 层、H=32、Dh=128、T=4096、FP16/BF16 ===")
    print(f"MHA KV Cache: {mha_bytes / 1024**3:.2f} GiB")
    print(f"MQA KV Cache: {mqa_bytes / 1024**2:.2f} MiB")
    print(f"KV 缩减: {mha_bytes / mqa_bytes:.0f}x")
    print(f"投影参数: MHA={mha_parameters:,}, MQA={mqa_parameters:,}")
    print(f"二者 attention FLOPs 均为: {flops / 1e9:.2f} GFLOPs/token")


if __name__ == "__main__":
    main()
