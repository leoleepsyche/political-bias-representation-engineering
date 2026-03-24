import sys
import types
import unittest
from unittest import mock

import torch

import political_dataset
import run_experiment


class FakeTokenizer:
    pad_token = None
    eos_token = "<eos>"


class FakeModel:
    def __init__(self, dtype):
        self.config = types.SimpleNamespace(num_hidden_layers=4)
        self._dtype = dtype
        self.moved_to = None
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self

    def parameters(self):
        yield types.SimpleNamespace(dtype=self._dtype)

    def to(self, device):
        self.moved_to = device
        return self


class LoadModelAndTokenizerTests(unittest.TestCase):
    def _install_fake_transformers(self, captured_kwargs, model_dtype=torch.float16):
        fake_transformers = types.ModuleType("transformers")

        class AutoTokenizer:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                return FakeTokenizer()

        class AutoModelForCausalLM:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                captured_kwargs.update(kwargs)
                return FakeModel(model_dtype)

        fake_transformers.AutoTokenizer = AutoTokenizer
        fake_transformers.AutoModelForCausalLM = AutoModelForCausalLM
        return fake_transformers

    def test_mps_auto_device_loads_without_device_map_and_moves_model(self):
        captured_kwargs = {}
        fake_transformers = self._install_fake_transformers(captured_kwargs)

        with mock.patch.dict(sys.modules, {"transformers": fake_transformers}):
            with mock.patch.object(run_experiment.torch.backends.mps, "is_available", return_value=True):
                with mock.patch.object(run_experiment.torch.cuda, "is_available", return_value=False):
                    model, tokenizer, device, num_layers = run_experiment.load_model_and_tokenizer(
                        "fake/model",
                        quantize=False,
                        device="auto",
                    )

        self.assertEqual(device, "mps")
        self.assertEqual(num_layers, 4)
        self.assertEqual(captured_kwargs["torch_dtype"], torch.float16)
        self.assertNotIn("device_map", captured_kwargs)
        self.assertEqual(model.moved_to, "mps")
        self.assertTrue(model.eval_called)
        self.assertEqual(tokenizer.pad_token, tokenizer.eos_token)

    def test_cpu_auto_device_uses_float32(self):
        captured_kwargs = {}
        fake_transformers = self._install_fake_transformers(captured_kwargs, model_dtype=torch.float32)

        with mock.patch.dict(sys.modules, {"transformers": fake_transformers}):
            with mock.patch.object(run_experiment.torch.backends.mps, "is_available", return_value=False):
                with mock.patch.object(run_experiment.torch.cuda, "is_available", return_value=False):
                    model, tokenizer, device, num_layers = run_experiment.load_model_and_tokenizer(
                        "fake/model",
                        quantize=False,
                        device="auto",
                    )

        self.assertEqual(device, "cpu")
        self.assertEqual(num_layers, 4)
        self.assertEqual(captured_kwargs["torch_dtype"], torch.float32)
        self.assertNotIn("device_map", captured_kwargs)
        self.assertIsNone(model.moved_to)
        self.assertTrue(model.eval_called)
        self.assertEqual(tokenizer.pad_token, tokenizer.eos_token)


class ParseArgsTests(unittest.TestCase):
    def test_parse_args_accepts_custom_output_dir(self):
        with mock.patch.object(
            sys,
            "argv",
            [
                "run_experiment.py",
                "--model",
                "Qwen/Qwen2.5-1.5B",
                "--output-dir",
                "results_base_1p5b",
            ],
        ):
            args = run_experiment.parse_args()

        self.assertEqual(args.model, "Qwen/Qwen2.5-1.5B")
        self.assertEqual(args.output_dir, "results_base_1p5b")


class PoliticalDatasetTests(unittest.TestCase):
    def test_get_right_statements_returns_topic_statement_pairs(self):
        right_statements = political_dataset.get_right_statements()
        first_topic, _, first_right_stmt = political_dataset.PAIRED_POLITICAL_STATEMENTS[0]

        self.assertTrue(all(len(item) == 2 for item in right_statements))
        self.assertEqual(right_statements[0][0], first_topic)
        self.assertEqual(right_statements[0][1], first_right_stmt)


if __name__ == "__main__":
    unittest.main()
