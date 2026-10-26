"""Export the repository's trained PyTorch checkpoints for browser inference.

Run from the repository root after installing requirements:
    python scripts/export_onnx.py --validate

The script deliberately fails rather than exporting randomly initialised weights.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.features.vocabulary import Vocabulary
from src.models.gru import SentimentGRU
from src.models.lstm import SentimentLSTM
from src.utils.config import Config
from src.utils.io import load_checkpoint


def export_one(name: str, model_cls: type[torch.nn.Module], config: Config, vocab: Vocabulary, output_dir: Path, validate: bool) -> None:
    checkpoint = config.artifacts_dir / f"{name}_best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Refusing export: trained checkpoint is missing: {checkpoint}")
    model = model_cls(len(vocab), config.embedding_dim, config.hidden_dim, config.pad_idx)
    load_checkpoint(checkpoint, model=model, map_location="cpu")
    model.eval()
    sample = torch.zeros((1, config.max_length), dtype=torch.long)
    target = output_dir / f"{name}.onnx"
    torch.onnx.export(
        model, sample, target, input_names=["input_ids"], output_names=["logits"],
        opset_version=17, do_constant_folding=True,
        dynamic_axes={"input_ids": {0: "batch"}, "logits": {0: "batch"}},
    )
    if validate:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("Install onnxruntime to validate exports: pip install onnxruntime") from exc
        sequence = torch.randint(0, len(vocab), (1, config.max_length), dtype=torch.long)
        with torch.no_grad():
            torch_logits = model(sequence).numpy()
        session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
        onnx_logits = session.run(["logits"], {"input_ids": sequence.numpy()})[0]
        if not torch.allclose(torch.from_numpy(onnx_logits), torch.from_numpy(torch_logits), rtol=1e-4, atol=1e-5):
            raise RuntimeError(f"ONNX parity validation failed for {name}")
    print(f"Exported {target} ({target.stat().st_size / 1024 / 1024:.2f} MiB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true", help="Compare ONNX Runtime logits to PyTorch logits.")
    args = parser.parse_args()
    config = Config()
    vocab = Vocabulary.load(config.artifacts_dir / "vocab.json")
    output_dir = Path("web/assets/models")
    output_dir.mkdir(parents=True, exist_ok=True)
    export_one("lstm", SentimentLSTM, config, vocab, output_dir, args.validate)
    export_one("gru", SentimentGRU, config, vocab, output_dir, args.validate)


if __name__ == "__main__":
    main()
