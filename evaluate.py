"""Standalone evaluation script for evaluating saved model checkpoints on test set."""

import argparse
from pathlib import Path
import torch

from src.data.dataset import IMDBDataset, create_dataloaders
from src.data.loader import load_imdb_csv, preprocess_dataframe, split_data
from src.evaluation.evaluator import evaluate_model
from src.evaluation.plots import plot_confusion_matrix_single
from src.features.vocabulary import Vocabulary
from src.models.gru import SentimentGRU
from src.models.lstm import SentimentLSTM
from src.utils.config import Config
from src.utils.io import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained sentiment model checkpoint.")
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["lstm", "gru"],
        help="Model architecture: 'lstm' or 'gru'.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to .pt checkpoint file (defaults to artifacts/{model_type}_best.pt).",
    )
    parser.add_argument(
        "--vocab-path",
        type=str,
        default=None,
        help="Path to vocab.json (defaults to artifacts/vocab.json).",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to IMDB Dataset.csv (defaults to data/IMDB Dataset.csv).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size (default: 64).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Logit decision threshold (default: 0.0, matching probability 0.5).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use ('cpu' or 'cuda').",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config()
    device = args.device or config.device

    ckpt_path = Path(args.checkpoint) if args.checkpoint else config.artifacts_dir / f"{args.model_type}_best.pt"
    vocab_path = Path(args.vocab_path) if args.vocab_path else config.artifacts_dir / "vocab.json"
    data_path = Path(args.data_path) if args.data_path else config.raw_data_path

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary file not found at: {vocab_path}")

    print(f"Loading vocabulary from: {vocab_path}")
    vocab = Vocabulary.load(vocab_path)

    print(f"Loading data from: {data_path}")
    raw_df = load_imdb_csv(data_path)
    clean_df = preprocess_dataframe(raw_df)
    _, _, test_df = split_data(
        clean_df,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        random_state=config.random_state,
    )
    print(f"Loaded test split: {len(test_df)} samples")

    test_dataset = IMDBDataset(
        reviews=test_df["review"],
        labels=test_df["label"],
        vocab=vocab,
        max_length=config.max_length,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model_class = SentimentLSTM if args.model_type == "lstm" else SentimentGRU
    model = model_class(
        vocab_size=len(vocab),
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        padding_idx=config.pad_idx,
    )

    print(f"Loading model weights from: {ckpt_path}")
    load_checkpoint(ckpt_path, model=model, map_location=device)

    metrics = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=device,
        model_name=args.model_type.upper(),
        threshold=args.threshold,
        verbose=True,
    )

    plot_confusion_matrix_single(
        metrics["confusion_matrix"],
        model_name=args.model_type.upper(),
        save_path=config.results_dir / f"eval_cm_{args.model_type}.png",
    )
    print(f"Evaluation complete. Confusion matrix saved to results/eval_cm_{args.model_type}.png")


if __name__ == "__main__":
    main()
