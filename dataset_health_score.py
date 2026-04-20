from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean, median


def gini(values: list[int]) -> float:
    n = len(values)
    if n == 0:
        return 0.0
    sorted_vals = sorted(values)
    total = sum(sorted_vals)
    if total == 0:
        return 0.0

    weighted_sum = 0
    for i, value in enumerate(sorted_vals, start=1):
        weighted_sum += i * value

    return (2 * weighted_sum) / (n * total) - (n + 1) / n


def grade(score: float) -> str:
    if score >= 85:
        return "A"
    if score >= 70:
        return "B"
    if score >= 55:
        return "C"
    if score >= 40:
        return "D"
    return "F"


def compute_score(counts: list[int], target_per_class: int, min_bootstrap_per_class: int) -> tuple[float, dict[str, float]]:
    if not counts:
        return 0.0, {
            "quantity": 0.0,
            "coverage20": 0.0,
            "coverage50": 0.0,
            "balance": 0.0,
            "min_support": 0.0,
        }

    class_count = len(counts)
    avg_count = mean(counts)
    min_count = min(counts)

    quantity = min(avg_count / float(target_per_class), 1.0)
    coverage20 = sum(1 for c in counts if c >= 20) / class_count
    coverage50 = sum(1 for c in counts if c >= 50) / class_count

    # 1.0 means perfectly balanced distribution.
    balance = 1.0 - gini(counts)

    # Penalize classes with very low floor counts.
    min_support = min(min_count / float(min_bootstrap_per_class), 1.0)

    # Weighted geometric-like blend with tiny floor values so the score is not locked at 0.
    # This keeps early-stage progress measurable while still penalizing weak dimensions.
    q = max(quantity, 0.01)
    c20 = max(coverage20, 0.01)
    c50 = max(coverage50, 0.01)
    bal = max(balance, 0.01)
    ms = max(min_support, 0.01)
    score = 100.0 * ((q ** 0.35) * (c20 ** 0.25) * (c50 ** 0.2) * (bal ** 0.1) * (ms ** 0.1))

    return score, {
        "quantity": quantity,
        "coverage20": coverage20,
        "coverage50": coverage50,
        "balance": balance,
        "min_support": min_support,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute dataset health/readiness score from class sample counts.")
    parser.add_argument("--data-dir", default="data", help="Path to dataset root directory.")
    parser.add_argument("--target-per-class", type=int, default=100, help="Target samples per class for robust training.")
    parser.add_argument("--min-bootstrap-per-class", type=int, default=10, help="Minimum sample floor per class for basic training.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise SystemExit(f"[ERROR] Data directory not found: {data_dir.resolve()}")

    class_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])
    counts: list[int] = []

    for class_dir in class_dirs:
        counts.append(len(list(class_dir.rglob("*.npy"))))

    total_classes = len(counts)
    total_samples = sum(counts)

    score, parts = compute_score(
        counts=counts,
        target_per_class=args.target_per_class,
        min_bootstrap_per_class=args.min_bootstrap_per_class,
    )

    print("=" * 62)
    print("Dataset Health Score")
    print("=" * 62)
    print(f"Data dir               : {data_dir.resolve()}")
    print(f"Total classes          : {total_classes}")
    print(f"Total samples          : {total_samples}")

    if counts:
        print(f"Samples/class (min)    : {min(counts)}")
        print(f"Samples/class (median) : {median(counts):.1f}")
        print(f"Samples/class (mean)   : {mean(counts):.2f}")
        print(f"Samples/class (max)    : {max(counts)}")

    print("-" * 62)
    print(f"Quantity score         : {parts['quantity'] * 100:6.2f} / 100")
    print(f"Coverage >=20          : {parts['coverage20'] * 100:6.2f} / 100")
    print(f"Coverage >=50          : {parts['coverage50'] * 100:6.2f} / 100")
    print(f"Balance score          : {parts['balance'] * 100:6.2f} / 100")
    print(f"Min support score      : {parts['min_support'] * 100:6.2f} / 100")
    print("-" * 62)
    print(f"Dataset Health Score   : {score:6.2f} / 100  (Grade: {grade(score)})")

    needed_20 = sum(max(0, 20 - c) for c in counts)
    needed_50 = sum(max(0, 50 - c) for c in counts)
    needed_100 = sum(max(0, 100 - c) for c in counts)

    print("-" * 62)
    print(f"Needed to reach >=20/class : {needed_20}")
    print(f"Needed to reach >=50/class : {needed_50}")
    print(f"Needed to reach >=100/class: {needed_100}")
    print("=" * 62)


if __name__ == "__main__":
    main()
