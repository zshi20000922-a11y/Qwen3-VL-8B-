import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ResultIntegrityTest(unittest.TestCase):
    def test_expected_scenarios_exist(self):
        expected = {f"resolution_{r}" for r in ("640x360", "1280x720", "1920x1080", "3840x2160")}
        expected |= {f"concurrency_{c}" for c in (1, 2, 4, 8)}
        for variant in ("vllm_bf16", "vllm_fp8"):
            data = json.loads((ROOT / f"results/{variant}/summary.json").read_text())
            self.assertEqual({row["scenario"] for row in data["scenarios"]}, expected)

    def test_raw_request_counts(self):
        cases = (("vllm_bf16", 95), ("vllm_fp8", 57), ("transformers_bf16", 20))
        for variant, expected in cases:
            lines = (ROOT / f"results/{variant}/requests.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), expected)

    def test_analysis_claims_are_consistent(self):
        data = json.loads((ROOT / "results/analysis/analysis.json").read_text())
        claims = data["fp8_vs_bf16"]
        self.assertGreater(claims["c8_aggregate_throughput_gain_pct"], 45)
        self.assertGreater(claims["weight_memory_reduction_pct"], 35)
        self.assertEqual(data["architecture"], "dense")


if __name__ == "__main__":
    unittest.main()
