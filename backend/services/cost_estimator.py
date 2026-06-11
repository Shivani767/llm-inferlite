from typing import Dict, Any

class CostEstimator:
    # Basic hourly-equivalent rates for local/on-prem research hardware (approximate)
    PRICES = {
        "local": {
            "rtx_3060": 0.12,
            "rtx_4090": 0.50,
            "a100_80gb": 3.67
        },
        "workstation": {
            "rtx_4090": 0.50,
            "a6000_48gb": 0.95
        },
        "lab": {
            "l40s_48gb": 1.25,
            "a100_80gb": 3.67
        }
    }

    def estimate_cost(self, model_size_gb: float, tps: float, monthly_requests: int, provider: str) -> Dict[str, Any]:
        # Simple estimation logic for local and lab-hosted hardware
        provider_key = provider.lower()
        instance = "rtx_3060" if model_size_gb < 16 else "rtx_4090"
        if provider_key == "lab" and model_size_gb >= 32:
            instance = "a100_80gb"
        elif provider_key == "workstation" and model_size_gb >= 32:
            instance = "a6000_48gb"

        hourly_rate = self.PRICES.get(provider_key, {}).get(instance, 1.0)
        
        # Assume 24/7 uptime for production-grade
        monthly_cost = hourly_rate * 24 * 30
        
        # Calculate cost per request
        total_tokens = monthly_requests * 500 # Avg tokens per request
        cost_per_request = monthly_cost / monthly_requests
        cost_per_1m_tokens = (monthly_cost / total_tokens) * 1_000_000
        
        return {
            "monthly_cost": monthly_cost,
            "cost_per_request": cost_per_request,
            "cost_per_1M_tokens": cost_per_1m_tokens,
            "recommended_setup": f"{provider_key.upper()} {instance}",
            "instance_details": {
                "instance_type": instance,
                "hourly_rate": hourly_rate
            }
        }
