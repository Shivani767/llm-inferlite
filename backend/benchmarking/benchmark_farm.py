from typing import List, Dict, Any
from database.repositories.model_repository import ModelRegistryRepository
from quantization.research_engines import QuantizationResearchManager
from evaluation.research_evaluator import ResearchEvaluator


class BenchmarkFarm:
    """
    Registry-triggered research pipeline. Quantization is measured or labeled unsupported.
    Quality suites that are not implemented (MMLU) stay unsupported.
    """

    def __init__(self, db_session):
        self.db = db_session
        self.registry = ModelRegistryRepository(db_session)
        self.quant_manager = QuantizationResearchManager()

    async def process_new_model(self, model_id: int, quant_methods: List[str] = None):
        model = self.registry.get_by_id(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")

        methods = quant_methods or ["fp32", "dynamic_int8", "int8_bnb", "awq", "gptq", "gguf"]
        results: Dict[str, Any] = {
            "model_name": model.name,
            "quantization_results": [],
            "evaluation_results": [],
        }

        evaluator = ResearchEvaluator(model.name, "base")
        results["base_metrics"] = (await evaluator.run_perplexity()).model_dump()
        results["mmlu"] = (await evaluator.run_mmlu()).model_dump()

        for method in methods:
            quant_res = self.quant_manager.run_quantization_experiment(method, model.name, {})
            results["quantization_results"].append(quant_res)
            results["evaluation_results"].append(
                {
                    "method": method,
                    "status": quant_res.get("status"),
                    "reason": quant_res.get("reason"),
                }
            )
        return results
