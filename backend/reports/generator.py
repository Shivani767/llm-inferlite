import datetime
from typing import Dict, Any, List
import json

class ResearchReportGenerator:
    """
    Module 14: Automated research report generation.
    Consolidates benchmarks, quantization deltas, and cost analysis.
    """
    
    def __init__(self, experiment_data: Dict[str, Any]):
        self.data = experiment_data
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def generate_markdown(self) -> str:
        """
        Creates a structured Markdown research report.
        """
        model_name = self.data.get("model_name", "LLM Experiment")
        report = f"# InferLite Research Report: {model_name}\n"
        report += f"**Date:** {self.timestamp}\n"
        report += f"**Hardware:** {self.data.get('hardware', 'Unknown')}\n\n"
        
        report += "## 1. Executive Summary\n"
        report += f"Analysis of {model_name} reveals that the optimal deployment strategy is "
        report += f"{self.data.get('optimal_config', 'yet to be determined')}.\n\n"
        
        report += "## 2. Performance Benchmarking\n"
        report += "| Runtime | Quantization | Latency (ms) | Throughput (TPS) | VRAM (GB) |\n"
        report += "| --- | --- | --- | --- | --- |\n"
        for res in self.data.get("benchmarks", []):
            report += f"| {res['runtime']} | {res['quant']} | {res['latency']} | {res['tps']} | {res['vram']} |\n"
        report += "\n"
        
        report += "## 3. Quantization Quality Analysis\n"
        report += "| Method | Perplexity Delta | Accuracy Retention | MSE Error |\n"
        report += "| --- | --- | --- | --- |\n"
        for q in self.data.get("quant_deltas", []):
            report += f"| {q['method']} | {q['ppl_delta']} | {q['retention']}% | {q['mse']} |\n"
        report += "\n"
        
        report += "## 4. Economic & Efficiency Analysis\n"
        report += f"- **Cost per 1M Tokens:** ${self.data.get('cost_per_1m', 0.0):.4f}\n"
        report += f"- **Energy Efficiency:** {self.data.get('tokens_per_joule', 0.0):.2f} Tokens/Joule\n"
        report += f"- **Carbon Footprint:** {self.data.get('co2_kg', 0.0):.4f} kg CO2 / month\n\n"
        
        report += "## 5. Deployment Recommendations\n"
        report += f"{self.data.get('recommendation_text', 'No recommendation available.')}\n"
            
        return report

    def export_to_json(self) -> str:
        return json.dumps({
            "metadata": {
                "engine": "InferLite-ReportGen-v0.1",
                "generated_at": self.timestamp
            },
            "payload": self.data
        }, indent=2)
