import unittest

from test import load_prompts_evaluate


class TestPromptLoad(unittest.TestCase):
    def test_prompts_evaluate_loaded(self):
        prompts = load_prompts_evaluate("prompt.txt")
        self.assertIsInstance(prompts, dict)
        self.assertTrue(any(v == "good" for v in prompts.values()))
        self.assertTrue(any(v == "bad" for v in prompts.values()))


if __name__ == "__main__":
    unittest.main()
