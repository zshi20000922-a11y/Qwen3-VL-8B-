import json, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

class ABResultsTest(unittest.TestCase):
    def test_cuda_graph_policy(self):
        d=json.loads((ROOT/'results/ab/comparison.json').read_text())
        rows={x['concurrency']:x for x in d['rows']}
        self.assertFalse(rows[1]['resume_eligible'])
        self.assertTrue(rows[8]['resume_eligible'])
        self.assertGreaterEqual(rows[8]['throughput_gain_pct'],5)
    def test_prefix_cache_not_claimed(self):
        d=json.loads((ROOT/'results/ab/prefix_comparison.json').read_text())
        self.assertTrue(all(abs(x['throughput_gain_pct'])<5 for x in d['rows']))

if __name__=='__main__': unittest.main()
