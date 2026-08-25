from typing import Dict, Any, List, Optional

from telemetry.gpu_engine import GPUTelemetryEngine


class EnergyIntelligenceService:
    def __init__(self):
        self.pue = 1.2
        self.carbon_intensity = 0.4
        self.telemetry = GPUTelemetryEngine()

    def get_gpu_power_usage(self) -> Dict[str, Any]:
        rows = self.telemetry.collect_metrics()
        if not rows:
            return {"status": "unsupported", "reason": "no telemetry"}
        row = rows[0]
        payload = row.model_dump()
        if row.status != "measured" or row.power_draw_w is None:
            payload["status"] = "unsupported"
            payload["reason"] = payload.get("reason") or "power_draw_w is not available without NVML"
        return payload

    def estimate_energy_cost(self, duration_seconds: float, avg_power_watts: float) -> Dict[str, Any]:
        """Uses caller-supplied power. Does not invent a wattage reading."""
        energy_kwh = (avg_power_watts * duration_seconds) / (3600 * 1000)
        total_energy_kwh = energy_kwh * self.pue
        return {
            "status": "derived",
            "note": "Derived from the provided avg_power_watts; not a GPU sensor reading.",
            "energy_consumption_kwh": total_energy_kwh,
            "carbon_footprint_kg_co2": total_energy_kwh * self.carbon_intensity,
            "pue": self.pue,
            "carbon_intensity_kg_per_kwh": self.carbon_intensity,
        }
