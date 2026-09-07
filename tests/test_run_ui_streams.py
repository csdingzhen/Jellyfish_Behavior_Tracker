import sys
import unittest

from scripts.run_ui import _ensure_standard_streams


class TestGuiStandardStreams(unittest.TestCase):
    def test_missing_streams_support_tqdm(self):
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        replacement_stdout = None
        replacement_stderr = None

        try:
            sys.stdout = None
            sys.stderr = None
            _ensure_standard_streams()

            replacement_stdout = sys.stdout
            replacement_stderr = sys.stderr
            self.assertTrue(callable(getattr(sys.stdout, "write", None)))
            self.assertTrue(callable(getattr(sys.stderr, "write", None)))

            from tqdm import tqdm

            self.assertEqual(list(tqdm(range(2))), [0, 1])
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            if replacement_stdout is not None:
                replacement_stdout.close()
            if replacement_stderr is not None:
                replacement_stderr.close()


if __name__ == "__main__":
    unittest.main()
