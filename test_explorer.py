"""Check the rendered explorer's data, correction notice and historical-series boundary."""
import csv
import io
import json
from pathlib import Path
import re
import unittest


class ExplorerTests(unittest.TestCase):
    def test_current_metrics_are_embedded_exactly(self):
        html = Path('index.html').read_text()
        embedded = re.search(r'const METRICS = (.*?);\n/\* METRICS-END', html)
        self.assertIsNotNone(embedded)
        self.assertEqual(json.loads(embedded[1]), json.loads(Path('content_metrics.json').read_text()))

    def test_each_truncation_point_matches_corrected_counts(self):
        html = Path('index.html').read_text()
        data = re.search(r'const DUMP_CSV = `(.*?)`;', html, re.S)[1]
        rows = list(csv.DictReader(io.StringIO(data)))
        with Path('corrected_by_dump.csv').open() as f:
            correct = {r['dump']: r for r in csv.DictReader(f)}
        self.assertEqual({r['dump'] for r in rows}, set(correct))
        for row in rows:
            expected = correct[row['dump']]
            self.assertEqual(int(row['truncated']), int(expected['truncated']))
            self.assertEqual(int(row['n_tok']), int(expected['n_tok']))
            self.assertAlmostEqual(float(row['trunc_pct']), int(row['truncated']) / int(row['n_tok']) * 100, places=2)

    def test_historical_series_is_not_active_and_notice_remains(self):
        html = Path('index.html').read_text()
        self.assertNotIn('FIXED_CSV', html)
        self.assertNotIn('58.2%', html)
        self.assertNotIn('const rows = fixed', html)
        self.assertIn('Fixed-score series: correction pending', html)
        self.assertIn('linked DOI and PDF remain the historical release', html)

    def test_dataset_card_body_is_generated_from_readme(self):
        card = Path('README_hf.md').read_text().split('---\n', 2)[2].lstrip('\n')
        self.assertEqual(card, Path('README.md').read_text())


if __name__ == '__main__':
    unittest.main()
