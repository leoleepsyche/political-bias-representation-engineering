import unittest
from pathlib import Path

import political_dataset_expanded


REPO_ROOT = Path(__file__).resolve().parent.parent


class ExpandedDatasetConsistencyTests(unittest.TestCase):
    def test_expanded_dataset_currently_contains_49_paired_topics(self):
        pairs = political_dataset_expanded.get_paired_statements()

        self.assertEqual(len(pairs), 49)
        self.assertEqual(len(political_dataset_expanded.get_left_statements()), 49)
        self.assertEqual(len(political_dataset_expanded.get_right_statements()), 49)
        self.assertEqual(len({topic for topic, _, _ in pairs}), 49)

    def test_step_1_report_matches_current_scope(self):
        report = (REPO_ROOT / "STEP_1_COMPLETION.md").read_text()

        self.assertIn("IN PROGRESS", report)
        self.assertIn("49 topics", report)
        self.assertNotIn("50 topics", report)
        self.assertIn("not yet satisfy the full Step 1 plan", report)


if __name__ == "__main__":
    unittest.main()
