#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


@dataclass
class TrainPoint:
    step: int
    loss: float
    lr: float
    tokens_per_sec: float


@dataclass
class EvalPoint:
    step: int
    op: str
    num_digits: int
    accuracy: float


def read_train_log(run_dir: str) -> List[TrainPoint]:
    path = os.path.join(run_dir, "train_log.csv")
    if not os.path.exists(path):
        return []
    points: List[TrainPoint] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                points.append(
                    TrainPoint(
                        step=int(row["step"]),
                        loss=float(row["loss"]),
                        lr=float(row["lr"]),
                        tokens_per_sec=float(row.get("tokens_per_sec", "nan")),
                    )
                )
            except Exception:
                continue
    points.sort(key=lambda p: p.step)
    return points


def read_eval_log(run_dir: str) -> List[EvalPoint]:
    path = os.path.join(run_dir, "eval_log.csv")
    if not os.path.exists(path):
        return []
    points: List[EvalPoint] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                points.append(
                    EvalPoint(
                        step=int(row["step"]),
                        op=str(row["op"]).strip(),
                        num_digits=int(row["num_digits"]),
                        accuracy=float(row["accuracy"]),
                    )
                )
            except Exception:
                continue
    points.sort(key=lambda p: (p.op, p.num_digits, p.step))
    return points


def find_eval_step_dirs(eval_root: str) -> List[str]:
    if not os.path.isdir(eval_root):
        return []
    dirs = []
    for name in os.listdir(eval_root):
        full = os.path.join(eval_root, name)
        if os.path.isdir(full) and name.startswith("step_"):
            dirs.append(full)
    dirs.sort(key=lambda p: int(os.path.basename(p).split("_")[1]))
    return dirs


def read_eval_details_for_step(step_dir: str) -> Dict[Tuple[str, int], Dict[str, float]]:
    results: Dict[Tuple[str, int], Dict[str, float]] = {}
    for op in ["+", "*"]:
        for num_digits in [1, 2, 3, 4, 5, 6]:
            fname = f"{op}_digits{num_digits}_details.csv"
            fpath = os.path.join(step_dir, fname)
            if not os.path.exists(fpath):
                alt = None
                for name in os.listdir(step_dir):
                    if name.endswith("_digits%d_details.csv" % num_digits) and name.startswith(op):
                        alt = os.path.join(step_dir, name)
                        break
                fpath = alt if alt and os.path.exists(alt) else None
            if not fpath:
                continue
            try:
                n = 0
                k = 0
                with open(fpath, newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        n += 1
                        try:
                            correct = int(row.get("correct", 0))
                        except Exception:
                            correct = 0
                        k += 1 if correct == 1 else 0
                if n > 0:
                    results[(op, num_digits)] = {"n": float(n), "k": float(k), "acc": (k / n)}
            except Exception:
                continue
    return results


def read_all_eval_details(eval_root: str) -> Dict[int, Dict[Tuple[str, int], Dict[str, float]]]:
    out: Dict[int, Dict[Tuple[str, int], Dict[str, float]]] = {}
    for step_dir in find_eval_step_dirs(eval_root):
        step_str = os.path.basename(step_dir).split("_")[1]
        try:
            step = int(step_str)
        except Exception:
            continue
        out[step] = read_eval_details_for_step(step_dir)
    return out


def wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    z = {
        0.80: 1.2815515655446004,
        0.90: 1.6448536269514722,
        0.95: 1.959963984540054,
        0.98: 2.3263478740408408,
        0.99: 2.5758293035489004,
    }.get(confidence, 1.959963984540054)
    phat = k / n
    denom = 1 + z * z / n
    center = phat + z * z / (2 * n)
    margin = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    lower = (center - margin) / denom
    upper = (center + margin) / denom
    return (max(0.0, lower), min(1.0, upper))


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def plot_training(points: List[TrainPoint], out_dir: str, logy: bool = False) -> None:
    if not HAS_MPL or not points:
        return
    steps = [p.step for p in points]
    loss = [p.loss for p in points]
    lr = [p.lr for p in points]
    tps = [p.tokens_per_sec for p in points]

    plt.figure(figsize=(9, 5))
    plt.plot(steps, loss, label="loss", color="#1f77b4")
    if logy:
        plt.yscale("log")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Training loss vs step")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "train_loss.png"))
    plt.close()

    plt.figure(figsize=(9, 3.8))
    plt.plot(steps, lr, label="lr", color="#ff7f0e")
    plt.xlabel("step")
    plt.ylabel("learning rate")
    plt.title("Learning rate schedule")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "lr_schedule.png"))
    plt.close()

    if all(not math.isnan(v) for v in tps):
        plt.figure(figsize=(9, 3.8))
        plt.plot(steps, tps, label="tokens/sec", color="#2ca02c")
        plt.xlabel("step")
        plt.ylabel("tokens/sec")
        plt.title("Throughput")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "tokens_per_sec.png"))
        plt.close()


def plot_eval_curves(points: List[EvalPoint], out_dir: str) -> None:
    if not HAS_MPL or not points:
        return
    grouped: Dict[Tuple[str, int], List[EvalPoint]] = defaultdict(list)
    for p in points:
        grouped[(p.op, p.num_digits)].append(p)
    # dynamic color mapping using a rotating palette
    palette = [
        "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b",
        "#e377c2","#7f7f7f","#bcbd22","#17becf","#aec7e8","#ffbb78",
        "#98df8a","#ff9896","#c5b0d5","#c49c94","#f7b6d2","#c7c7c7",
        "#dbdb8d","#9edae5",
    ]
    keys = sorted(grouped.keys(), key=lambda k: (k[0], k[1]))
    color_map = {k: palette[i % len(palette)] for i, k in enumerate(keys)}

    plt.figure(figsize=(11, 6))
    for key in keys:
        series = sorted(grouped[key], key=lambda p: p.step)
        steps = [p.step for p in series]
        acc = [p.accuracy for p in series]
        label = f"{key[0]} digits={key[1]}"
        plt.plot(steps, acc, label=label, color=color_map[key])
    plt.xlabel("step")
    plt.ylabel("accuracy")
    plt.ylim(0.0, 1.0)
    plt.title("Eval accuracy over steps")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "eval_accuracy.png"))
    plt.close()


def plot_macro_average(points: List[EvalPoint], out_dir: str) -> None:
    if not HAS_MPL or not points:
        return
    by_step: Dict[int, List[float]] = defaultdict(list)
    for p in points:
        by_step[p.step].append(p.accuracy)
    steps = sorted(by_step.keys())
    macro = [sum(by_step[s]) / len(by_step[s]) for s in steps]
    plt.figure(figsize=(9, 4))
    plt.plot(steps, macro, color="#1f77b4")
    plt.xlabel("step")
    plt.ylabel("macro avg acc")
    plt.ylim(0.0, 1.0)
    plt.title("Macro-average accuracy over tasks")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "macro_average.png"))
    plt.close()


def get_last_by_task(points: List[EvalPoint]) -> Dict[Tuple[str, int], EvalPoint]:
    by_task: Dict[Tuple[str, int], List[EvalPoint]] = defaultdict(list)
    for p in points:
        by_task[(p.op, p.num_digits)].append(p)
    last: Dict[Tuple[str, int], EvalPoint] = {}
    for k, v in by_task.items():
        v_sorted = sorted(v, key=lambda p: p.step)
        last[k] = v_sorted[-1]
    return last


def plot_last_step_bars(points: List[EvalPoint], out_dir: str) -> None:
    if not HAS_MPL or not points:
        return
    last = get_last_by_task(points)
    tasks = sorted(last.keys(), key=lambda k: (k[0], k[1]))
    accs = [last[k].accuracy for k in tasks]
    labels = [f"{k[0]} {k[1]}" for k in tasks]
    plt.figure(figsize=(11, 4))
    bars = plt.bar(range(len(tasks)), accs, color="#4e79a7")
    plt.xticks(range(len(tasks)), labels, rotation=0)
    plt.ylim(0, 1)
    plt.ylabel("accuracy")
    plt.title("Last-step accuracy by task")
    for i, b in enumerate(bars):
        h = b.get_height()
        plt.text(b.get_x() + b.get_width()/2, h + 0.01, f"{h:.2f}", ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "last_step_bars.png"))
    plt.close()


def plot_last_step_heatmap(points: List[EvalPoint], out_dir: str) -> None:
    if not HAS_MPL or not points:
        return
    last = get_last_by_task(points)
    ops = sorted({op for op, _ in last.keys()})
    digits = sorted({d for _, d in last.keys()})
    grid = []
    for op in ops:
        row = []
        for d in digits:
            ep = last.get((op, d))
            row.append(ep.accuracy if ep else float('nan'))
        grid.append(row)
    plt.figure(figsize=(1.2*len(digits) + 2, 0.8*len(ops) + 2))
    im = plt.imshow(grid, vmin=0.0, vmax=1.0, cmap='viridis', aspect='auto')
    plt.colorbar(im, fraction=0.046, pad=0.04, label='accuracy')
    plt.xticks(range(len(digits)), [str(d) for d in digits])
    plt.yticks(range(len(ops)), ops)
    plt.xlabel("digits")
    plt.title("Last-step accuracy heatmap")
    # annotate
    for i in range(len(ops)):
        for j in range(len(digits)):
            val = grid[i][j]
            if not math.isnan(val):
                plt.text(j, i, f"{val:.2f}", ha='center', va='center', color='white' if val < 0.5 else 'black', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "last_step_heatmap.png"))
    plt.close()


def write_eval_summary(
    eval_points: List[EvalPoint],
    detail_stats: Dict[int, Dict[Tuple[str, int], Dict[str, float]]],
    out_dir: str,
    eval_max_examples: Optional[int] = None,
) -> Tuple[str, Dict[Tuple[str, int], Dict[str, float]]]:
    by_task: Dict[Tuple[str, int], List[EvalPoint]] = defaultdict(list)
    for p in eval_points:
        by_task[(p.op, p.num_digits)].append(p)

    summary: Dict[Tuple[str, int], Dict[str, float]] = {}
    rows: List[List[str]] = [[
        "op",
        "num_digits",
        "last_step",
        "last_acc",
        "best_step",
        "best_acc",
        "delta_best_last",
        "n_last",
        "ci95_lower",
        "ci95_upper",
    ]]

    for task, series in by_task.items():
        series_sorted = sorted(series, key=lambda p: p.step)
        last = series_sorted[-1]
        best = max(series_sorted, key=lambda p: p.accuracy)
        n_last = None
        ci_lower = float("nan")
        ci_upper = float("nan")
        if last.step in detail_stats and task in detail_stats[last.step]:
            n = int(detail_stats[last.step][task].get("n", 0))
            k = int(round(detail_stats[last.step][task].get("k", 0)))
            if n > 0:
                n_last = n
                ci_lower, ci_upper = wilson_ci(k, n, 0.95)
        elif eval_max_examples:
            n_last = eval_max_examples
            k_est = int(round(last.accuracy * n_last))
            ci_lower, ci_upper = wilson_ci(k_est, n_last, 0.95)
        summary[task] = {
            "last_step": float(last.step),
            "last_acc": float(last.accuracy),
            "best_step": float(best.step),
            "best_acc": float(best.accuracy),
            "delta_best_last": float(last.accuracy - best.accuracy),
            "n_last": float(n_last) if n_last is not None else float("nan"),
            "ci95_lower": float(ci_lower),
            "ci95_upper": float(ci_upper),
        }
        rows.append([
            task[0],
            str(task[1]),
            str(last.step),
            f"{last.accuracy:.4f}",
            str(best.step),
            f"{best.accuracy:.4f}",
            f"{(last.accuracy - best.accuracy):+.4f}",
            str(n_last if n_last is not None else ""),
            f"{ci_lower:.4f}",
            f"{ci_upper:.4f}",
        ])

    csv_path = os.path.join(out_dir, "eval_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    md_lines: List[str] = []
    md_lines.append("# Eval Summary\n")
    md_lines.append("\n")
    md_lines.append("| op | digits | last step | last acc | best step | best acc | Δ(last-best) | n@last | 95% CI |\n")
    md_lines.append("|---:|:------:|---------:|---------:|---------:|---------:|------------:|------:|:-----:|\n")
    for task in sorted(summary.keys(), key=lambda k: (k[0], k[1])):
        s = summary[task]
        md_lines.append(
            f"| {task[0]} | {task[1]} | {int(s['last_step'])} | {s['last_acc']:.4f} | {int(s['best_step'])} | {s['best_acc']:.4f} | {s['delta_best_last']:+.4f} | {'' if math.isnan(s['n_last']) else int(s['n_last'])} | [{s['ci95_lower']:.3f}, {s['ci95_upper']:.3f}] |\n"
        )
    md_lines.append("\n")
    md_lines.append("Notes: CI uses Wilson score; n comes from details when available, else eval_max_examples.\n")
    md_path = os.path.join(out_dir, "eval_summary.md")
    with open(md_path, "w") as f:
        f.writelines(md_lines)

    return md_path, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a training run: plots + eval summaries")
    parser.add_argument("--run_dir", type=str, default=os.path.abspath(os.path.join(os.path.dirname(__file__), "runs/20250908_154411")), help="Path to run directory (containing train_log.csv, eval_log.csv, eval/)")
    parser.add_argument("--logy", action="store_true", help="Plot training loss on a log scale")
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    eval_root = os.path.join(run_dir, "eval")
    out_dir = os.path.join(run_dir, "analysis")
    ensure_dir(out_dir)

    hparams_path = os.path.join(run_dir, "hparams.json")
    eval_max_examples = None
    if os.path.exists(hparams_path):
        try:
            with open(hparams_path) as f:
                hp = json.load(f)
                if isinstance(hp, dict) and "eval_max_examples" in hp:
                    eval_max_examples = int(hp["eval_max_examples"])  # type: ignore
        except Exception:
            pass

    train_points = read_train_log(run_dir)
    eval_points = read_eval_log(run_dir)
    detail_stats = read_all_eval_details(eval_root)

    plot_training(train_points, out_dir, logy=args.logy)
    plot_eval_curves(eval_points, out_dir)
    plot_macro_average(eval_points, out_dir)
    plot_last_step_bars(eval_points, out_dir)
    plot_last_step_heatmap(eval_points, out_dir)

    md_path, summary = write_eval_summary(eval_points, detail_stats, out_dir, eval_max_examples)

    json_path = os.path.join(out_dir, "eval_summary.json")
    with open(json_path, "w") as f:
        json.dump({str(k): v for k, v in summary.items()}, f, indent=2)

    print(f"Wrote analysis to: {out_dir}")
    if HAS_MPL:
        print("Generated plots: train_loss.png, lr_schedule.png, tokens_per_sec.png (if available), eval_accuracy.png, macro_average.png, last_step_bars.png, last_step_heatmap.png")
    else:
        print("matplotlib not available; skipped plots. Install matplotlib to enable plotting.")
    print(f"Eval summary: {md_path}")


if __name__ == "__main__":
    main()

