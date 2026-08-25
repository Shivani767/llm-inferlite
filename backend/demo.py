"""Demo: environment, capabilities, and a tiny in-memory measured benchmark."""

from research.capabilities import probe
from research.engine import run_benchmark
from research.env import collect_environment
from research.testing import TinyTokenizer, make_tiny_lm


def run_demo():
    print("=== InferLite research demo (measured, not simulated) ===\n")
    env = collect_environment()
    print("[1] Environment")
    print(f"  python={env.get('python')} system={env.get('system')} machine={env.get('machine')}")
    print(f"  torch={env.get('packages', {}).get('torch')} cuda={env.get('torch', {}).get('cuda_available')}")

    caps = probe()
    print("\n[2] Capabilities")
    print(f"  device={caps['device']}")
    for name, item in caps["experiments"].items():
        flag = "yes" if item["supported"] else "no"
        print(f"  [{flag}] {name}: {item['reason']}")

    print("\n[3] In-memory tiny model benchmark (real wall-clock, not GPT-scale)")
    rec = run_benchmark(
        model_id="tiny-in-memory",
        method="fp32",
        prompt="hello",
        max_new_tokens=4,
        warmup_runs=0,
        measure_runs=2,
        model=make_tiny_lm(),
        tokenizer=TinyTokenizer(),
    )
    print(f"  status={rec.status.value}")
    if rec.metrics and rec.metrics.ttft_ms:
        print(f"  TTFT mean ms={rec.metrics.ttft_ms.mean:.2f}")
        print(f"  load_time_s={rec.metrics.load_time_s:.4f}")
    print("\n=== Done. Use `python -m research suite --config ../configs/macbook_cpu.yaml` for HF models. ===")


if __name__ == "__main__":
    run_demo()
