import unittest
from exo.models import get_supported_models, model_cards
from exo.inference.inference_engine import inference_engine_classes
from typing import NamedTuple

class TestCase(NamedTuple):
  name: str
  engine_lists: list  # Will contain short names, will be mapped to class names
  expected_models_contains: list
  min_count: int | None
  exact_count: int | None
  max_count: int | None

# Helper function to map short names to class names
def expand_engine_lists(engine_lists):
  def map_engine(engine):
    return inference_engine_classes.get(engine, engine)  # Return original name if not found

  return [[map_engine(engine) for engine in sublist]
          for sublist in engine_lists]

test_cases = [
  TestCase(
    name="single_mlx_engine",
    engine_lists=[["mlx"]],
    expected_models_contains=["llama-3.2-1b", "llama-3.1-70b", "mistral-nemo"],
    min_count=10,
    exact_count=None,
    max_count=None
  ),
  TestCase(
    name="single_tinygrad_engine",
    engine_lists=[["tinygrad"]],
    expected_models_contains=["llama-3.2-1b", "llama-3.2-3b"],
    min_count=5,
    exact_count=None,
    max_count=15
  ),
  TestCase(
    name="multiple_engines_or",
    engine_lists=[["mlx", "tinygrad"], ["mlx"]],
    expected_models_contains=["llama-3.2-1b", "llama-3.2-3b", "mistral-nemo"],
    min_count=10,
    exact_count=None,
    max_count=None
  ),
  TestCase(
    name="multiple_engines_all",
    engine_lists=[["mlx", "tinygrad"], ["mlx", "tinygrad"]],
    expected_models_contains=["llama-3.2-1b", "llama-3.2-3b", "mistral-nemo"],
    min_count=10,
    exact_count=None,
    max_count=None
  ),
  TestCase(
    name="distinct_engine_lists",
    engine_lists=[["mlx"], ["tinygrad"]],
    expected_models_contains=["llama-3.2-1b"],
    min_count=5,
    exact_count=None,
    max_count=15
  ),
  TestCase(
    name="no_engines",
    engine_lists=[],
    expected_models_contains=None,
    min_count=None,
    exact_count=len(model_cards),
    max_count=None
  ),
  TestCase(
    name="nonexistent_engine",
    engine_lists=[["NonexistentEngine"]],
    expected_models_contains=[],
    min_count=None,
    exact_count=0,
    max_count=None
  ),
  TestCase(
    name="dummy_engine",
    engine_lists=[["dummy"]],
    expected_models_contains=["dummy"],
    min_count=None,
    exact_count=1,
    max_count=None
  ),
]

class TestModelHelpers(unittest.TestCase):
  def test_get_supported_models(self):
    for case in test_cases:
      with self.subTest(f"{case.name}_short_names"):
        result = get_supported_models(case.engine_lists)
        self._verify_results(case, result)

      with self.subTest(f"{case.name}_class_names"):
        class_name_lists = expand_engine_lists(case.engine_lists)
        result = get_supported_models(class_name_lists)
        self._verify_results(case, result)

  def _verify_results(self, case, result):
    if case.expected_models_contains:
      for model in case.expected_models_contains:
        self.assertIn(model, result)

    if case.min_count:
      self.assertGreater(len(result), case.min_count)

    if case.exact_count is not None:
      self.assertEqual(len(result), case.exact_count)

    # Special case for distinct lists test
    if case.name == "distinct_engine_lists":
      self.assertLess(len(result), 15)
      self.assertNotIn("mistral-nemo", result)

    if case.max_count:
      self.assertLess(len(result), case.max_count)

if __name__ == '__main__':
  unittest.main()
