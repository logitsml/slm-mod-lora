from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .model import WarmStartModerator


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _parse_label(value: str) -> int:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "removed", "remove", "yes"}:
        return 1
    if normalized in {"0", "false", "kept", "keep", "no"}:
        return 0
    raise ValueError(f"Unsupported label value: {value!r}")


def train(args: argparse.Namespace) -> None:
    rows = _read_csv(Path(args.csv))
    comments = [row[args.text_col] for row in rows]
    labels = [_parse_label(row[args.label_col]) for row in rows]

    model = WarmStartModerator(
        encoder_name=args.encoder,
        prefix=args.prefix,
        batch_size=args.batch_size,
        device=args.device,
    )
    model.fit(comments, labels)
    model.save(args.output)
    print(f"saved {args.output}")


def score(args: argparse.Namespace) -> None:
    rows = _read_csv(Path(args.csv))
    comments = [row[args.text_col] for row in rows]
    model = WarmStartModerator.load(args.model)
    probabilities = model.predict_proba(comments)
    decisions = (probabilities >= args.threshold).astype(int)

    fieldnames = list(rows[0].keys()) + ["removal_probability", "remove_decision"] if rows else [
        "removal_probability",
        "remove_decision",
    ]
    with Path(args.output).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row, proba, decision in zip(rows, probabilities, decisions):
            out = dict(row)
            out["removal_probability"] = f"{proba:.8f}"
            out["remove_decision"] = str(int(decision))
            writer.writerow(out)
    print(f"saved {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="modmodmod-warmstart")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="train a warm-start moderation head")
    train_parser.add_argument("--csv", required=True)
    train_parser.add_argument("--text-col", default="body")
    train_parser.add_argument("--label-col", default="label")
    train_parser.add_argument("--output", required=True)
    train_parser.add_argument("--encoder", default="intfloat/e5-large-v2")
    train_parser.add_argument("--prefix", default="query: ")
    train_parser.add_argument("--batch-size", type=int, default=128)
    train_parser.add_argument("--device", default="auto")
    train_parser.set_defaults(func=train)

    score_parser = subparsers.add_parser("score", help="score comments with a trained warm-start head")
    score_parser.add_argument("--model", required=True)
    score_parser.add_argument("--csv", required=True)
    score_parser.add_argument("--text-col", default="body")
    score_parser.add_argument("--output", required=True)
    score_parser.add_argument("--threshold", type=float, default=0.5)
    score_parser.set_defaults(func=score)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
