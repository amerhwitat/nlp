import tempfile
import unittest
from pathlib import Path

from iso_tool.pipeline import BuildPipeline, BuildProgress


class PipelineResilienceTests(unittest.TestCase):
    def test_parallel_continues_after_runtime_error_and_reports_it(self):
        with tempfile.TemporaryDirectory() as td:
            pipeline = BuildPipeline(Path(td), workers=2)
            events = []

            def good():
                return "good"

            def bad():
                raise RuntimeError("simulated compiler failure")

            results = pipeline.parallel(
                [bad, good],
                progress=events.append,
                on_error=lambda exc, index: events.append(
                    BuildProgress("error", index, 2, str(exc))
                ),
            )

            self.assertEqual(len(results), 2)
            self.assertTrue(any(r.ok is False for r in results))
            self.assertTrue(any(r.ok and r.value == "good" for r in results))
            self.assertTrue(any(e.stage == "error" for e in events))
            self.assertEqual(events[-1].completed, 2)

    def test_run_safe_returns_failure_instead_of_raising(self):
        with tempfile.TemporaryDirectory() as td:
            pipeline = BuildPipeline(Path(td))
            result = pipeline.run_safe(["definitely-not-a-real-iso-tool-command"], timeout=1)
            self.assertFalse(result.ok)
            self.assertTrue(result.error)


if __name__ == "__main__":
    unittest.main()
