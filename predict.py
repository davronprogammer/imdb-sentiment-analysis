"""Inference script for single reviews or interactive sentiment prediction."""

import argparse
from pathlib import Path
import torch

from src.features.encoding import encode_text
from src.features.tokenizer import clean_text
from src.features.vocabulary import Vocabulary
from src.models.gru import SentimentGRU
from src.models.lstm import SentimentLSTM
from src.utils.config import Config
from src.utils.io import load_checkpoint


def predict_sentiment(
    text: str,
    model: torch.nn.Module,
    vocab: Vocabulary,
    max_length: int = 200,
    device: str = "cpu",
    threshold: float = 0.0,
):
    """Predict sentiment (label, confidence probability) for a raw input string.

    Args:
        text: Raw user review text.
        model: Trained SentimentLSTM or SentimentGRU.
        vocab: Vocabulary instance.
        max_length: Sequence max length.
        device: 'cpu' or 'cuda'.
        threshold: Logit classification threshold (default: 0.0).

    Returns:
        Dictionary with cleaned text, raw logit, probability, predicted label, and label name.
    """
    model.eval()
    cleaned = clean_text(text)
    encoded = encode_text(
        text=cleaned,
        vocab=vocab,
        max_length=max_length,
        pad_idx=vocab.pad_idx,
        unk_idx=vocab.unk_idx,
    )

    tensor = torch.tensor([encoded], dtype=torch.long, device=device)

    with torch.no_grad():
        logit = model(tensor).item()
        prob = torch.sigmoid(torch.tensor(logit)).item()

    is_positive = logit >= threshold
    label_str = "Positive" if is_positive else "Negative"

    return {
        "raw_text": text,
        "cleaned_text": cleaned,
        "logit": round(logit, 4),
        "probability": round(prob, 4),
        "prediction": int(is_positive),
        "sentiment": label_str,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Predict sentiment for a movie review.")
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Review text to classify. If omitted, runs in interactive prompt mode.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="lstm",
        choices=["lstm", "gru"],
        help="Model architecture to use: 'lstm' or 'gru' (default: lstm).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint .pt file.",
    )
    parser.add_argument(
        "--vocab-path",
        type=str,
        default=None,
        help="Path to vocab.json.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use ('cpu' or 'cuda').",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config()
    device = args.device

    ckpt_path = Path(args.checkpoint) if args.checkpoint else config.artifacts_dir / f"{args.model_type}_best.pt"
    vocab_path = Path(args.vocab_path) if args.vocab_path else config.artifacts_dir / "vocab.json"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}. Please train the model first.")
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary not found at: {vocab_path}. Please run training first.")

    vocab = Vocabulary.load(vocab_path)

    model_class = SentimentLSTM if args.model_type == "lstm" else SentimentGRU
    model = model_class(
        vocab_size=len(vocab),
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        padding_idx=config.pad_idx,
    )
    load_checkpoint(ckpt_path, model=model, map_location=device)
    model.to(device)

    if args.text:
        result = predict_sentiment(args.text, model, vocab, config.max_length, device)
        print(f"\nReview: '{args.text}'")
        print(f"Prediction: {result['sentiment']} (Confidence: {result['probability'] * 100:.1f}%, Logit: {result['logit']})")
    else:
        print(f"\nLoaded {args.model_type.upper()} sentiment classifier.")
        print("Enter review text to predict sentiment (or 'exit' to quit):")
        while True:
            try:
                line = input("\nReview > ")
                if not line or line.strip().lower() in ["exit", "quit", "q"]:
                    break
                res = predict_sentiment(line, model, vocab, config.max_length, device)
                print(f"Sentiment: {res['sentiment']} (Confidence: {res['probability'] * 100:.1f}%, Logit: {res['logit']})")
            except (KeyboardInterrupt, EOFError):
                break


if __name__ == "__main__":
    main()
