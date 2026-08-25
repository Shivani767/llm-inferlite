"""Speculative decoding lab wrapping the measured greedy implementation."""

from typing import Dict, Any, List

from research.experiments.speculative import run_speculative_suite


class SpeculativeDecodingLab:
    def __init__(self, target_model_id: str, draft_model_id: str):
        self.target_model_id = target_model_id
        self.draft_model_id = draft_model_id

    def run_speculative_experiment(self, gamma: int = 4) -> Dict[str, Any]:
        recs = run_speculative_suite(
            self.target_model_id,
            self.draft_model_id,
            gammas=[gamma],
            max_new_tokens=16,
            measure_runs=1,
        )
        spec = next((r for r in recs if "speculative" in r.method), recs[-1] if recs else None)
        return spec.model_dump() if spec else {"status": "error", "reason": "no records"}

    def compare_strategies(self) -> List[Dict[str, Any]]:
        recs = run_speculative_suite(
            self.target_model_id,
            self.draft_model_id,
            gammas=[4],
            max_new_tokens=16,
            measure_runs=1,
        )
        return [r.model_dump() for r in recs]
