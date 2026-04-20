from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, median


def read_counts(data_dir: Path) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    for class_dir in sorted([p for p in data_dir.iterdir() if p.is_dir()]):
        count = len(list(class_dir.rglob("*.npy")))
        rows.append((class_dir.name, count))
    return rows


def write_csv(rows: list[tuple[str, int, int]], out_file: Path) -> None:
    with out_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_name", "current_count", "needed_to_target"])
        writer.writerows(rows)


def write_task_markdown(
    rows: list[tuple[str, int, int]],
    out_file: Path,
    people: int,
    daily_target_per_person: int,
    target: int,
) -> None:
    assignments: list[list[tuple[str, int, int]]] = [[] for _ in range(max(1, people))]
    for i, row in enumerate(rows):
        assignments[i % len(assignments)].append(row)

    total_needed = sum(r[2] for r in rows)
    throughput = max(1, people * daily_target_per_person)
    est_days = (total_needed + throughput - 1) // throughput

    lines: list[str] = []
    lines.append("# Data Collection Handoff Plan")
    lines.append("")
    lines.append(f"Target per class: {target}")
    lines.append(f"Total extra samples needed: {total_needed}")
    lines.append(f"Team size: {people}")
    lines.append(f"Daily throughput target: {throughput} samples/day")
    lines.append(f"Estimated duration: {est_days} days")
    lines.append("")
    lines.append("## Team Assignments")
    lines.append("")

    for idx, bucket in enumerate(assignments, start=1):
        bucket_needed = sum(r[2] for r in bucket)
        lines.append(f"### Person {idx} (needed: {bucket_needed})")
        for class_name, current, needed in bucket:
            lines.append(f"- [ ] {class_name}: current={current}, add={needed}")
        lines.append("")

    out_file.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan class-based data collection backlog and teammate assignments.")
    parser.add_argument("--data-dir", default="data", help="Dataset root directory.")
    parser.add_argument("--target", type=int, default=20, help="Target sample count per class.")
    parser.add_argument("--people", type=int, default=2, help="Number of teammates collecting data.")
    parser.add_argument("--daily-target-per-person", type=int, default=80, help="Expected daily samples per person.")
    parser.add_argument("--top", type=int, default=80, help="Only include worst N classes (lowest sample counts).")
    parser.add_argument("--csv-out", default="collection_backlog.csv", help="CSV output file.")
    parser.add_argument("--md-out", default="handoff_tasks.md", help="Markdown task plan output file.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise SystemExit(f"[ERROR] Data directory not found: {data_dir.resolve()}")

    counts = read_counts(data_dir)
    if not counts:
        raise SystemExit("[ERROR] No class folders found.")

    sorted_low = sorted(counts, key=lambda x: (x[1], x[0]))
    selected = sorted_low[: max(1, args.top)]

    backlog: list[tuple[str, int, int]] = []
    for class_name, current in selected:
        needed = max(0, args.target - current)
        backlog.append((class_name, current, needed))

    write_csv(backlog, Path(args.csv_out))
    write_task_markdown(
        backlog,
        Path(args.md_out),
        people=max(1, args.people),
        daily_target_per_person=max(1, args.daily_target_per_person),
        target=max(1, args.target),
    )

    values = [c for _, c in counts]
    total_classes = len(values)
    total_samples = sum(values)
    print("=" * 64)
    print("Collection Backlog Planner")
    print("=" * 64)
    print(f"Data dir               : {data_dir.resolve()}")
    print(f"Total classes          : {total_classes}")
    print(f"Total samples          : {total_samples}")
    print(f"Samples/class min      : {min(values)}")
    print(f"Samples/class median   : {median(values):.1f}")
    print(f"Samples/class mean     : {mean(values):.2f}")
    print(f"Samples/class max      : {max(values)}")
    print("-" * 64)
    print(f"Planning target/class  : {args.target}")
    print(f"Planned class subset   : {len(backlog)}")
    print(f"Subset backlog total   : {sum(x[2] for x in backlog)}")
    print(f"CSV written            : {Path(args.csv_out).resolve()}")
    print(f"Task plan written      : {Path(args.md_out).resolve()}")
    print("=" * 64)


if __name__ == "__main__":
    main()
