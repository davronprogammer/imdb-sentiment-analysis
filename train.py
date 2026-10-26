"""Reproducible training and evaluation workflow for IMDB Sentiment Classification (LSTM vs GRU)."""

import argparse
from pathlib import Path
import pandas as pd
import torch

from src.data.dataset import create_dataloaders
from src.data.loader import load_imdb_csv, preprocess_dataframe, split_data
from src.evaluation.evaluator import evaluate_model
from src.evaluation.plots import (
    plot_confusion_matrix_single,
    plot_side_by_side_confusion_matrices,
    plot_training_history,
)
from src.features.vocabulary import Vocabulary
from src.models.base import count_parameters
from src.models.gru import SentimentGRU
from src.models.lstm import SentimentLSTM
from src.training.trainer import Trainer
from src.utils.config import Config
from src.utils.io import save_json
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate LSTM and GRU sentiment models.")
    parser.add_argument(
        "--models",
        type=str,
        default="both",
        choices=["lstm", "gru", "both"],
        help="Model architecture(s) to train: 'lstm', 'gru', or 'both' (default: both).",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default: 5).")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64).")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001).")
    parser.add_argument(
        "--patience",
        type=int,
        default=2,
        help="Early stopping patience epochs on validation loss (default: 2, 0 to disable).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of total samples for rapid debugging/diagnostic runs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use ('cpu', 'cuda'). Defaults to CUDA if available, else CPU.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42).")
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to IMDB Dataset.csv (default: data/IMDB Dataset.csv).",
    )
    return parser.parse_args()


def run_pipeline(args):
    config = Config()
    set_seed(args.seed)

    device = args.device or config.device
    print(f"Using device: {device}")

    data_path = Path(args.data_path) if args.data_path else config.raw_data_path
    print(f"Loading data from: {data_path}")

    raw_df = load_imdb_csv(data_path)
    print(f"Raw dataset shape: {raw_df.shape}")

    if args.max_samples and len(raw_df) > args.max_samples:
        print(f"Subsampling to {args.max_samples} samples for diagnostic run...")
        raw_df = raw_df.sample(n=args.max_samples, random_state=args.seed).reset_index(drop=True)

    print("Preprocessing and cleaning reviews...")
    clean_df = preprocess_dataframe(raw_df)
    print(f"Cleaned dataset shape: {clean_df.shape} (removed {len(raw_df) - len(clean_df)} duplicates)")

    print("Splitting dataset into train (80%), validation (10%), and test (10%)...")
    train_df, val_df, test_df = split_data(
        clean_df,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        random_state=args.seed,
    )
    print(f"Split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Vocabulary construction (training data only)
    vocab_path = config.artifacts_dir / "vocab.json"
    print(f"Building vocabulary (max size={config.max_vocab_size}) from train split only...")
    vocab = Vocabulary.build_from_texts(
        texts=train_df["review"],
        max_size=config.max_vocab_size,
        pad_token=config.pad_token,
        unk_token=config.unk_token,
        pad_idx=config.pad_idx,
        unk_idx=config.unk_idx,
    )
    vocab.save(vocab_path)
    print(f"Vocabulary built: {len(vocab)} tokens (saved to {vocab_path})")

    # DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        vocab=vocab,
        batch_size=args.batch_size,
        max_length=config.max_length,
    )

    models_to_train = []
    if args.models in ["lstm", "both"]:
        models_to_train.append(("LSTM", SentimentLSTM))
    if args.models in ["gru", "both"]:
        models_to_train.append(("GRU", SentimentGRU))

    training_histories = {}
    eval_results = {}
    comparison_records = []

    for model_name, model_class in models_to_train:
        print(f"\n{'#' * 50}")
        print(f" Starting training: {model_name}")
        print(f"{'#' * 50}")

        model = model_class(
            vocab_size=len(vocab),
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            padding_idx=config.pad_idx,
        )

        param_count = count_parameters(model)
        print(f"{model_name} trainable parameters: {param_count:,}")

        trainer = Trainer(
            model=model,
            device=device,
            learning_rate=args.lr,
            checkpoint_dir=config.artifacts_dir,
            early_stopping_patience=args.patience,
        )

        fit_res = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            model_name=model_name.lower(),
            extra_metadata={
                "vocab_size": len(vocab),
                "embedding_dim": config.embedding_dim,
                "hidden_dim": config.hidden_dim,
                "batch_size": args.batch_size,
                "max_length": config.max_length,
            },
            verbose=True,
        )

        training_histories[model_name] = fit_res["history"]

        # Evaluate best checkpoint on test set
        print(f"\nEvaluating best {model_name} on held-out test set...")
        metrics = evaluate_model(
            model=model,
            dataloader=test_loader,
            device=device,
            model_name=model_name,
            threshold=0.0,
            verbose=True,
        )
        eval_results[model_name] = metrics

        # Save classification report
        report_file = config.results_dir / f"classification_report_{model_name.lower()}.txt"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(f"{model_name} Classification Report\n")
            f.write("=" * 40 + "\n\n")
            f.write(metrics["classification_report"])

        comparison_records.append(
            {
                "Model": model_name,
                "Test Accuracy": round(metrics["accuracy"], 4),
                "Positive F1": round(metrics["positive_f1"], 4),
                "Macro F1": round(metrics["macro_f1"], 4),
                "Negative Precision": round(metrics["negative_precision"], 4),
                "Negative Recall": round(metrics["negative_recall"], 4),
                "Positive Precision": round(metrics["positive_precision"], 4),
                "Positive Recall": round(metrics["positive_recall"], 4),
                "Trainable Parameters": param_count,
                "Training Time (s)": round(fit_res["training_time"], 2),
                "Evaluation Time (s)": round(metrics["evaluation_time"], 2),
                "Best Epoch": fit_res["best_epoch"],
                "Best Val Loss": round(fit_res["best_val_loss"], 4),
                "Best Val Accuracy": round(fit_res["best_val_acc"], 4),
            }
        )

    # Save Comparison CSV & JSON
    comp_df = pd.DataFrame(comparison_records)
    csv_path = config.results_dir / "model_comparison.csv"
    comp_df.to_csv(csv_path, index=False)
    print(f"\nComparison table saved to: {csv_path}")

    json_path = config.results_dir / "model_comparison.json"
    save_json(comparison_records, json_path)

    # Save training history CSV
    history_rows = []
    for m_name, h in training_histories.items():
        for ep in range(len(h["train_loss"])):
            history_rows.append(
                {
                    "model": m_name,
                    "epoch": ep + 1,
                    "train_loss": h["train_loss"][ep],
                    "val_loss": h["val_loss"][ep],
                    "train_acc": h["train_acc"][ep],
                    "val_acc": h["val_acc"][ep],
                }
            )
    hist_df = pd.DataFrame(history_rows)
    hist_csv_path = config.results_dir / "training_history.csv"
    hist_df.to_csv(hist_csv_path, index=False)
    print(f"Training history saved to: {hist_csv_path}")

    # Save evaluation results JSON
    save_json(eval_results, config.results_dir / "evaluation_results.json")

    # Generate and save plots
    if training_histories:
        plot_training_history(
            training_histories,
            save_path=config.results_dir / "training_history.png",
        )
        print("Training curves plot saved to results/training_history.png")

    cms = {name: res["confusion_matrix"] for name, res in eval_results.items()}
    if len(cms) > 1:
        plot_side_by_side_confusion_matrices(
            cms,
            save_path=config.results_dir / "confusion_matrices.png",
        )
        print("Side-by-side confusion matrix plot saved to results/confusion_matrices.png")

    for name, cm in cms.items():
        plot_confusion_matrix_single(
            cm,
            model_name=name,
            save_path=config.results_dir / f"confusion_matrix_{name.lower()}.png",
        )

    # Concise experiment summary
    summary_txt = config.results_dir / "experiment_summary.txt"
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("IMDB Sentiment Classification Experiment Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(comp_df.to_string(index=False))
        f.write("\n\nNotes:\n")
        f.write(f"- Device: {device}\n")
        f.write(f"- Random seed: {args.seed}\n")
        f.write(f"- Epochs configured: {args.epochs}\n")
        f.write(f"- Batch size: {args.batch_size}\n")
        f.write(f"- Vocabulary size: {len(vocab)}\n")

    print("\nSummary of Run:")
    print(comp_df.to_string(index=False))
    print("\nTraining and evaluation pipeline finished successfully!")


if __name__ == "__main__":
    cli_args = parse_args()
    run_pipeline(cli_args)
