import click
import requests
import json
import time

API_URL = "http://localhost:8003/api/v1"

def print_table(headers, rows):
    """Simple replacement for tabulate to avoid dependency issues."""
    if not rows:
        return
    
    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(str(val)))
            
    # Print headers
    header_str = " | ".join(f"{h:<{widths[i]}}" for i, h in enumerate(headers))
    click.echo(header_str)
    click.echo("-" * len(header_str))
    
    # Print rows
    for row in rows:
        row_str = " | ".join(f"{str(v):<{widths[i]}}" for i, v in enumerate(row))
        click.echo(row_str)

@click.group()
def cli():
    """InferLite CLI: Research and Optimization Tool for LLMs."""
    pass

@cli.command()
def health():
    """Check the health of the InferLite API."""
    click.echo(f"Checking health at {API_URL}...")
    try:
        response = requests.get(f"{API_URL.replace('/api/v1', '')}/health")
        if response.status_code == 200:
            click.echo(click.style("Γ£ô InferLite API is healthy", fg="green"))
        else:
            click.echo(click.style(f"Γ£û API returned status {response.status_code}", fg="red"))
    except Exception as e:
        click.echo(click.style(f"Γ£û Could not connect to API: {e}", fg="red"))

@cli.command()
@click.option("--rps", default=10.0, help="Requests per second")
@click.option("--duration", default=10, help="Duration in seconds")
@click.option("--nodes", default=1, help="Number of serving nodes")
def simulate(rps, duration, nodes):
    """Run a serving architecture simulation."""
    click.echo(f"Starting simulation: {rps} RPS for {duration}s on {nodes} nodes...")
    
    payload = {
        "rps": rps,
        "duration_seconds": duration,
        "nodes": nodes,
        "burstiness": 0.2,
        "concurrency_limit": 32,
        "latency_p50_ms": 200,
        "latency_p99_ms": 500,
        "tokens_per_request": 128
    }
    
    try:
        response = requests.post(f"{API_URL}/simulator/simulate", json=payload)
        response.raise_for_status()
        res = response.json()
        
        rows = [
            ["Total Requests", res["total_requests"]],
            ["Successful", res["successful_requests"]],
            ["Avg Latency (ms)", f"{res['avg_latency_ms']:.2f}"],
            ["P99 Latency (ms)", f"{res['p99_latency_ms']:.2f}"],
            ["Throughput (RPS)", f"{res['throughput_rps']:.2f}"],
            ["Max Concurrency", res["max_concurrency"]]
        ]
        click.echo("")
        print_table(["Metric", "Value"], rows)
    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg="red"))

@cli.command()
def energy():
    """Get real-time GPU energy telemetry."""
    try:
        response = requests.get(f"{API_URL}/energy/gpu/telemetry")
        response.raise_for_status()
        data = response.json()
        
        if "error" in data:
            click.echo(click.style(f"Warning: {data['error']}", fg="yellow"))
            return

        rows = [
            ["Power Draw (W)", f"{data['power_draw_watts']:.2f}"],
            ["Temperature (C)", f"{data['temperature_c']:.2f}"],
            ["Fan Speed (%)", f"{data['fan_speed_pct']:.2f}"],
            ["Memory Used (MB)", f"{data['memory_used_mb']:.2f}"]
        ]
        click.echo("")
        print_table(["Telemetry", "Value"], rows)
    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg="red"))

@cli.command()
@click.argument("model_path")
def analyze(model_path):
    """Analyze an ONNX model for optimizations."""
    click.echo(f"Analyzing model at {model_path}...")
    try:
        response = requests.post(f"{API_URL}/lab/onnx/analyze", params={"model_path": model_path})
        response.raise_for_status()
        data = response.json()
        
        click.echo("\nOperator Counts:")
        op_rows = [[op, count] for op, count in data.get("op_counts", {}).items()]
        print_table(["Operator", "Count"], op_rows)
        
        click.echo(f"\nTotal Nodes: {data['total_nodes']}")
        click.echo(f"Optimization Score: {data.get('optimization_score', 'N/A')}")
    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg="red"))

@cli.command()
@click.option("--model", default="llama-3-8b", help="Model name")
@click.option("--hardware", default="A100-80GB", help="Hardware profile")
def recommend(model, hardware):
    """Get AI-driven optimization recommendations."""
    try:
        params = {
            "model_name": model,
            "hardware": hardware,
            "latency_sla": 200,
            "budget": 1000.0
        }
        response = requests.get(f"{API_URL}/advisor/recommend", params=params)
        response.raise_for_status()
        r = response.json()
        
        click.echo("\nOptimal Recommendation:")
        click.echo(click.style(f"Γû║ Runtime: {r['best_runtime']}", fg="blue", bold=True))
        click.echo(f"  Quantization: {r['best_quantization']}")
        click.echo(f"  Batch Size: {r['best_batch_size']}")
        click.echo(f"  Cache Strategy: {r['best_cache_strategy']}")
        click.echo(f"  Expected TPS: {r['expected_tps']}")
        click.echo(f"  Expected Memory: {r['expected_memory_gb']} GB")
        click.echo(f"\n  Rationale: {r['explanation']}")
    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg="red"))

if __name__ == "__main__":
    cli()
