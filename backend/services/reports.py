import datetime
from typing import Dict, Any, List

class ResearchReportGenerator:
    """
    Generates comprehensive research reports in Markdown or PDF format.
    """
    
    def __init__(self, experiment_data: Dict[str, Any]):
        self.data = experiment_data
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def generate_markdown(self) -> str:
        """
        Creates a structured Markdown report of the experiment results.
        """
        report = f"# InferLite Research Report: {self.data.get('model_name', 'LLM Experiment')}\n"
        report += f"**Date:** {self.timestamp}\n\n"
        
        report += "## 1. Executive Summary\n"
        report += f"This report analyzes the performance and quality trade-offs for {self.data.get('model_name')}. "
        report += "Key findings indicate that quantization reduced memory by X% with only Y% perplexity increase.\n\n"
        
        report += "## 2. Benchmark Results\n"
        report += "| Metric | Value | Delta |\n"
        report += "| --- | --- | --- |\n"
        for metric, val in self.data.get("metrics", {}).items():
            report += f"| {metric} | {val} | - |\n"
        report += "\n"
        
        report += "## 3. Environmental Impact\n"
        energy = self.data.get("energy", {})
        report += f"- **Energy Consumed:** {energy.get('kwh', 0):.4f} kWh\n"
        report += f"- **Carbon Footprint:** {energy.get('co2_kg', 0):.4f} kg CO2\n\n"
        
        report += "## 4. Recommendations\n"
        for rec in self.data.get("recommendations", []):
            report += f"- **{rec['type'].capitalize()}:** {rec['reason']}\n"
            
        return report

    def export_to_json(self) -> Dict[str, Any]:
        return {
            "metadata": {"generated_at": self.timestamp},
            "report_content": self.data
        }
