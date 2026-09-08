import argparse
import csv
from datetime import datetime
from pathlib import Path

import torch

from mha_mqa_lab.mha import mha_attention
from mha_mqa_lab.mqa import mqa_attention


MIB = 1024**2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark MHA/MQA decode.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--contexts", default="128,512,2048,8192")
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--max-memory-mib", type=float, default=512)
    return parser.parse_args()


def time_attention(operation, repeats: int) -> tuple[float, float, float]:
    for _ in range(10):
        operation()
    torch.cuda.synchronize()

    events = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        events.append((start, end))
    torch.cuda.synchronize()

    samples = torch.tensor(
        [start.elapsed_time(end) * 1_000 for start, end in events]
    )
    p20, median, p80 = torch.quantile(samples, torch.tensor([0.2, 0.5, 0.8]))
    return p20.item(), median.item(), p80.item()


def run_case(
    variant: str,
    batch: int,
    heads: int,
    head_dim: int,
    context: int,
    repeats: int,
) -> tuple[float, float, float, float]:
    kv_heads = heads if variant == "mha" else 1
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    query = torch.randn(batch, heads, 1, head_dim,
                        device="cuda", dtype=torch.float16)
    key = torch.randn(
        batch,
        kv_heads,
        context,
        head_dim,
        device="cuda",
        dtype=torch.float16,
    )
    value = torch.randn_like(key)
    operation = (
        (lambda: mha_attention(query, key, value))
        if variant == "mha"
        else (lambda: mqa_attention(query, key, value))
    )
    p20, median, p80 = time_attention(operation, repeats)
    peak_mib = (torch.cuda.max_memory_allocated() - before) / MIB
    return p20, median, p80, peak_mib


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    contexts = [int(value) for value in args.contexts.split(",")]
    torch.manual_seed(42)
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    memory_limit = min(args.max_memory_mib * MIB, free_bytes * 0.2)
    print(f"GPU: {torch.cuda.get_device_name()} ({total_bytes / MIB:.0f} MiB)")
    print(f"Memory guard: {memory_limit / MIB:.0f} MiB\n")
    print("     T  type   KV MiB  median us  logical GB/s  speedup")

    rows = []
    for context in contexts:
        pair = []
        for variant in ("mha", "mqa"):
            kv_heads = args.num_heads if variant == "mha" else 1
            cache_bytes = (
                2
                * args.batch_size
                * context
                * kv_heads
                * args.head_dim
                * 2
            )
            estimated_bytes = 1.25 * (
                cache_bytes + 8 * args.batch_size * args.num_heads * context
            )
            if estimated_bytes > memory_limit:
                print(f"{context:>6} {variant.upper():>5}  skipped by memory guard")
                continue
            try:
                p20, median, p80, peak_mib = run_case(
                    variant,
                    args.batch_size,
                    args.num_heads,
                    args.head_dim,
                    context,
                    args.repeats,
                )
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"{context:>6} {variant.upper():>5}  skipped after CUDA OOM")
                continue
            row = {
                "gpu": torch.cuda.get_device_name(),
                "dtype": "float16",
                "batch_size": args.batch_size,
                "num_heads": args.num_heads,
                "head_dim": args.head_dim,
                "context": context,
                "variant": variant,
                "kv_cache_mib": cache_bytes / MIB,
                "p20_us": p20,
                "median_us": median,
                "p80_us": p80,
                "logical_kv_gb_s": cache_bytes / (median / 1e6) / 1e9,
                "peak_mib": peak_mib,
                "speedup_vs_mha": "",
            }
            pair.append(row)
            torch.cuda.empty_cache()

        if len(pair) == 2:
            pair[0]["speedup_vs_mha"] = 1.0
            pair[1]["speedup_vs_mha"] = (
                pair[0]["median_us"] / pair[1]["median_us"]
            )
        for row in pair:
            speedup = row["speedup_vs_mha"]
            speedup_text = f"{speedup:.2f}" if speedup != "" else "-"
            print(
                f"{context:>6} {row['variant'].upper():>5} "
                f"{row['kv_cache_mib']:>8.2f} {row['median_us']:>10.2f} "
                f"{row['logical_kv_gb_s']:>13.2f} {speedup_text:>8}"
            )
        rows.extend(pair)

    if rows:
        output = Path("benchmark-results/mha-mqa")
        output.mkdir(parents=True, exist_ok=True)
        csv_path = output / f"{datetime.now():%Y%m%d-%H%M%S}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults saved to: {csv_path.resolve()}")


if __name__ == "__main__":
    main()
