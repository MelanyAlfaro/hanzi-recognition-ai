from src.training.classifier import Classifier
from src.training.binary_relevance import BinaryRelevanceClassifier
from src.training.unified import UnifiedClassifier
from src.tools.radical_dataset import RadicalDataset
from src.tools.metrics import evaluate_predictions
from typing import Type
import argparse
from pathlib import Path
import config
import json
from datetime import datetime
import time
import numpy as np
from typing import Any


def test(
  classifier: Classifier,
  test_ds: RadicalDataset
) -> tuple[np.ndarray, np.ndarray, float]:
  start = time.time()
  images = test_ds.x.float().div_(255.0)

  y_true = test_ds.y.numpy().astype(int)

  probs = classifier.predict_proba(images).numpy()
  y_pred = (probs >= config.THRESHOLD).astype(int)

  end = time.time()
  return y_true, y_pred, end - start


def round_metrics(metrics: dict) -> dict:
  for key, value in metrics.items():
    if isinstance(value, float):
      metrics[key] = round(value, 4)
    elif isinstance(value, list):
      metrics[key] = [round(n, 4) if isinstance(n, float) else n for n in value]
    else:
      metrics[key] = value
  return metrics


def log_test(
  y_true: np.ndarray,
  y_pred: np.ndarray,
  classifier_type: str,
  duration: float,
  log_dir: Path
) -> None:
  log_file = log_dir / f"{classifier_type}-{datetime.now()}.json"
  metrics = evaluate_predictions(y_true, y_pred)
  metrics = round_metrics(metrics)
  log = json.dumps(
    {
      "classifier_type": classifier_type,
      "time": duration,
      "metrics:": metrics
    },
    indent=2
  )
  log_file.write_text(log)
  print(f"Saved to {log_file}")


def laod_classifier(weights: Path, n_labels: int, classifier: Type) -> Classifier:
  """Create and load a classifier.

  Args:
    weights: The file where the weights are.
    n_labels: The number of radicals to test.
    classifier: The type of classifier to construct, either
      BinaryRelevanceClassifier or UnifiedClassifier.

  Returns:
    The classifier that is created.
  """
  cls = classifier(n_labels)
  cls.load(weights)
  return cls

def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--out", type=Path, default=Path(config.LOGS_DIR),
                      help="output folder (default: config.LOGS_DIR)")
  parser.add_argument("--model", type=Path, default=Path(config.RESULTS_DIR),
                      help="input for the models (default: config.RESULTS_DIR)")
  args = parser.parse_args()

  args.out.mkdir(parents=True, exist_ok=True)

  test_ds = RadicalDataset(split="test")
  n_labels = test_ds.n_radicals

  binary_path = args.model / f"{config.BINARY}.pt"
  unified_path = args.model / f"{config.UNIFIED}.pt"

  binary = laod_classifier(binary_path, n_labels, BinaryRelevanceClassifier)
  unified = laod_classifier(unified_path, n_labels, UnifiedClassifier)

  print("Starting tests for binary relavance classifier")
  y_true_binary, y_pred_binary, binary_time = test(binary, test_ds)
  print("Finished tests for binary relavance classifier")

  print("Starting tests for unified classifier")
  y_true_unified, y_pred_unified, unified_time = test(unified, test_ds)
  print("Finished tests for unified classifier")

  log_test(y_true_binary, y_pred_binary, config.BINARY, binary_time, args.out)
  log_test(y_true_unified, y_pred_unified, config.UNIFIED, unified_time, args.out)


if __name__ == "__main__":
  main()