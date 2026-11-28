"""Train a model with the dataset."""

import config
import argparse
import json
import time
from pathlib import Path
from src.tools.radical_dataset import RadicalDataset
from src.training.classifier import Classifier
from src.training.binary_relevance import BinaryRelevanceClassifier
from src.training.unified import UnifiedClassifier

def train(
  classifier: Classifier,
  train_ds: RadicalDataset,
  val_ds: RadicalDataset,
  out_dir: Path,
  classifier_type: str
) -> None:
  print(f"Training {classifier_type} classifier")

  start = time.time()
  classifier.fit(train_ds=train_ds, val_ds=val_ds)
  end = time.time()
  elapsed = end - start

  print(f"Trained in {elapsed:.5}s")

  weights_path = out_dir / f"{classifier_type}.pt"
  log_path = out_dir / f"{classifier_type}.json"

  classifier.save(weights_path)

  log_text = json.dumps(
    {
      "model": classifier_type,
      "n_labels": classifier.n_labels,
      "n_params": classifier.count_parameters(),
      "train_seconds": round(elapsed, 4),
      "weights": str(weights_path),
      "config": {
        "epochs": config.EPOCHS, "batch_size": config.BATCH_SIZE,
        "lr": config.LEARNING_RATE, "img_size": config.IMG_SIZE,
        "seed": config.SEED, "threshold": config.THRESHOLD,
      },
    },
    indent=2
  )

  log_path.write_text(log_text)

  print(f"Weights visualization -> {weights_path}")
  print(f"Creation log          -> {log_path}")


def train_binary(
  n_labels: int,
  train_ds: RadicalDataset,
  val_ds: RadicalDataset,
  out_dir: Path
) -> None:
  config.apply_seed()
  train(
    BinaryRelevanceClassifier(n_labels),
    train_ds,
    val_ds,
    out_dir,
    config.BINARY
  )


def train_unified(
  n_labels: int,
  train_ds: RadicalDataset,
  val_ds: RadicalDataset,
  out_dir: Path
) -> None:
  config.apply_seed()
  train(
    UnifiedClassifier(n_labels),
    train_ds,
    val_ds,
    out_dir,
    config.UNIFIED
  )


def train_all(
  n_labels: int,
  train_ds: RadicalDataset,
  val_ds: RadicalDataset,
  out_dir: Path
) -> None:
  train_binary(n_labels, train_ds, val_ds, out_dir)
  train_unified(n_labels, train_ds, val_ds, out_dir)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", choices=["all", config.BINARY, config.UNIFIED], default="all",
                      help="Type of classifier to train: unified, binary, or all")
  parser.add_argument("--out", type=Path, default=Path(config.RESULTS_DIR),
                      help="output folder (default: config.RESULTS_DIR)")
  parser.add_argument("--data", type=Path, default=Path(config.PROCESSED_DIR),
                      help="folder containing the processed data (default: config.PROCESSED_DIR)")
  args = parser.parse_args()

  args.out.mkdir(parents=True, exist_ok=True)

  train_ds = RadicalDataset(split="train")
  val_ds = RadicalDataset(split="val")
  n_labels = train_ds.n_radicals

  if args.model == "all": train_all(n_labels, train_ds, val_ds, args.out)
  elif args.model == config.UNIFIED: train_unified(n_labels, train_ds, val_ds, args.out)
  elif args.model == config.BINARY: train_binary(n_labels, train_ds, val_ds, args.out)


if __name__ == "__main__":
  main()
