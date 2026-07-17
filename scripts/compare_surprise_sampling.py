from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import yaml


STRATEGIES = [
    "uniform",
    "soft_surprise",
    "warmup_soft_surprise",
    "warmup_learnable_surprise",
    "capped_learnable_surprise",
]

GROUPS = ["easy", "hard", "noise"]

COLORS = {
    "uniform": "#2563eb",
    "soft_surprise": "#dc2626",
    "warmup_soft_surprise": "#059669",
    "warmup_learnable_surprise": "#7c3aed",
    "capped_learnable_surprise": "#ea580c",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return

    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def has_split(rows: list[dict[str, str]], split: str) -> bool:
    return any(row.get("split") == split for row in rows)


def find_latest_runs(root: Path) -> list[dict[str, object]]:
    candidates: dict[str, list[dict[str, object]]] = defaultdict(list)

    for status_path in sorted(root.glob("*/status.json")):
        run_dir = status_path.parent
        status = json.loads(status_path.read_text())
        config = status.get("config", {})
        strategy = config.get("sampling_strategy")

        if strategy not in STRATEGIES:
            continue

        summary_path = run_dir / "surprise_summary.csv"
        sampling_path = run_dir / "sampling_history.csv"

        if not summary_path.exists() or not sampling_path.exists():
            continue

        summary = read_csv(summary_path)

        if not has_split(summary, "test"):
            continue

        candidates[strategy].append(
            {
                "strategy": strategy,
                "run_dir": run_dir,
                "summary": summary,
                "sampling": read_csv(sampling_path),
                "status": status,
                "mtime": status_path.stat().st_mtime,
            }
        )

    runs = []
    for strategy in STRATEGIES:
        if strategy in candidates:
            runs.append(sorted(candidates[strategy], key=lambda x: float(x["mtime"]))[-1])

    return runs


def row_x(row: dict[str, str]) -> int:
    if "step" in row and row["step"] != "":
        return int(float(row["step"]))
    return int(float(row["epoch"]))


def split_group_points(
    run: dict[str, object],
    split: str,
    group: str,
    field: str,
) -> list[tuple[int, float]]:
    result = []
    for row in run["summary"]:
        if row["split"] == split and row["group"] == group:
            result.append((row_x(row), float(row[field])))
    return sorted(result)


def sampling_points(
    run: dict[str, object],
    group: str,
) -> list[tuple[int, float]]:
    result = []
    for row in run["sampling"]:
        if row["group"] == group:
            result.append((row_x(row), float(row["selected_fraction"])))
    return sorted(result)


def make_panel_curves(
    runs: list[dict[str, object]],
    kind: str,
    split: str | None,
    group: str,
) -> dict[str, list[tuple[int, float]]]:
    curves = {}
    for run in runs:
        strategy = str(run["strategy"])
        if kind == "summary":
            assert split is not None
            curves[strategy] = split_group_points(
                run,
                split,
                group,
                "raw_mse_surprise_mean",
            )
        elif kind == "sampling":
            curves[strategy] = sampling_points(run, group)
        else:
            raise ValueError(kind)
    return curves


def dashboard_svg(
    path: Path,
    title: str,
    subtitle: str,
    panels: list[tuple[str, dict[str, list[tuple[int, float]]]]],
    y_label: str,
) -> None:
    width = 1380
    panel_height = 285
    height = 95 + panel_height * len(panels) + 35

    left = 95.0
    plot_w = 900.0
    plot_h = 190.0
    legend_x = left + plot_w + 55

    all_points = [
        point
        for _, curves in panels
        for points in curves.values()
        for point in points
    ]

    if not all_points:
        path.write_text("")
        return

    x_min = min(x for x, _ in all_points)
    x_max = max(x for x, _ in all_points)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        f'<text x="32" y="42" font-family="Arial" font-size="25" font-weight="700" fill="#0f172a">{title}</text>',
        f'<text x="32" y="68" font-family="Arial" font-size="15" fill="#475569">{subtitle}</text>',
    ]

    for panel_index, (panel_title, curves) in enumerate(panels):
        panel_top = 95.0 + panel_index * panel_height
        plot_top = panel_top + 35.0

        panel_points = [point for points in curves.values() for point in points]
        if not panel_points:
            continue

        y_min = min(y for _, y in panel_points)
        y_max = max(y for _, y in panel_points)
        if y_max <= y_min:
            y_min -= 0.1
            y_max += 0.1
        else:
            pad = 0.1 * (y_max - y_min)
            y_min -= pad
            y_max += pad

        def xy(x_value: int, y_value: float) -> tuple[float, float]:
            x = left + (x_value - x_min) / max(x_max - x_min, 1) * plot_w
            y = plot_top + plot_h - (y_value - y_min) / max(y_max - y_min, 1e-12) * plot_h
            return x, y

        parts.append(
            f'<text x="32" y="{panel_top + 10}" font-family="Arial" font-size="18" font-weight="700" fill="#0f172a">{panel_title}</text>'
        )

        for i in range(5):
            frac = i / 4
            y = plot_top + plot_h - frac * plot_h
            value = y_min + frac * (y_max - y_min)
            parts.append(f'<line x1="{left}" x2="{left + plot_w}" y1="{y:.2f}" y2="{y:.2f}" stroke="#e2e8f0"/>')
            parts.append(f'<text x="30" y="{y + 4:.2f}" font-family="Arial" font-size="11" fill="#334155">{value:.4g}</text>')

        parts.append(
            f'<line x1="{left}" x2="{left + plot_w}" y1="{plot_top + plot_h}" y2="{plot_top + plot_h}" stroke="#94a3b8"/>'
        )

        for i in range(5):
            frac = i / 4
            x = left + frac * plot_w
            step = round(x_min + frac * (x_max - x_min))
            parts.append(
                f'<text x="{x:.2f}" y="{plot_top + plot_h + 23}" font-family="Arial" font-size="11" fill="#334155" text-anchor="middle">{step}</text>'
            )

        for strategy in STRATEGIES:
            points = curves.get(strategy, [])
            if not points:
                continue

            d = []
            for j, (x_value, y_value) in enumerate(points):
                x, y = xy(x_value, y_value)
                d.append(f'{"M" if j == 0 else "L"} {x:.2f} {y:.2f}')

            color = COLORS[strategy]
            parts.append(
                f'<path d="{" ".join(d)}" fill="none" stroke="{color}" stroke-width="2.2"/>'
            )

        if panel_index == 0:
            for index, strategy in enumerate(STRATEGIES):
                y = plot_top + index * 24
                color = COLORS[strategy]
                parts.append(
                    f'<line x1="{legend_x}" x2="{legend_x + 30}" y1="{y}" y2="{y}" stroke="{color}" stroke-width="2.4"/>'
                )
                parts.append(
                    f'<text x="{legend_x + 42}" y="{y + 5}" font-family="Arial" font-size="14" fill="#0f172a">{strategy}</text>'
                )

    parts.append(
        f'<text x="{left + plot_w / 2}" y="{height - 18}" font-family="Arial" font-size="13" fill="#334155" text-anchor="middle">optimizer step</text>'
    )
    parts.append("</svg>")
    path.write_text("".join(parts))


def final_split_metrics(
    runs: list[dict[str, object]],
    split: str,
) -> list[dict[str, object]]:
    result = []

    for run in runs:
        strategy = str(run["strategy"])
        rows = [
            row for row in run["summary"]
            if row["split"] == split and row["group"] in GROUPS
        ]
        max_step = max(row_x(row) for row in rows)
        final_rows = [row for row in rows if row_x(row) == max_step]

        out = {
            "strategy": strategy,
            "split": split,
            "final_step": max_step,
        }

        raw_values = []

        for row in final_rows:
            group = row["group"]
            raw = float(row["raw_mse_surprise_mean"])
            lp = float(row["positive_learning_progress_mean"])
            sampling_score = float(row["sampling_score_mean"])

            out[f"{group}_raw_mse"] = raw
            out[f"{group}_positive_learning_progress"] = lp
            out[f"{group}_sampling_score"] = sampling_score
            raw_values.append(raw)

        out["overall_raw_mse"] = sum(raw_values) / len(raw_values)
        result.append(out)

    return result


def final_sampling_metrics(runs: list[dict[str, object]]) -> list[dict[str, object]]:
    result = []

    for run in runs:
        strategy = str(run["strategy"])
        rows = run["sampling"]
        max_step = max(row_x(row) for row in rows)
        window_start = max_step - 10_000
        window = [row for row in rows if row_x(row) >= window_start]

        by_group = defaultdict(list)
        for row in window:
            if row["group"] in GROUPS:
                by_group[row["group"]].append(float(row["selected_fraction"]))

        out = {
            "strategy": strategy,
            "final_step": max_step,
        }

        for group in GROUPS:
            values = by_group[group]
            out[f"selected_{group}_mean_last_window"] = sum(values) / len(values) if values else 0.0

        result.append(out)

    return result


def trapezoid_mean(points: list[tuple[int, float]], max_step: int) -> float:
    points = [(x, y) for x, y in sorted(points) if x <= max_step]

    if not points:
        return float("nan")

    if len(points) == 1:
        return points[0][1]

    area = 0.0
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        area += (x1 - x0) * (y0 + y1) / 2.0

    span = points[-1][0] - points[0][0]
    if span <= 0:
        return points[-1][1]

    return area / span


def final_group_value(
    run: dict[str, object],
    split: str,
    group: str,
    field: str,
) -> tuple[int, float]:
    rows = [
        row for row in run["summary"]
        if row["split"] == split and row["group"] == group
    ]
    max_step = max(row_x(row) for row in rows)
    final_rows = [row for row in rows if row_x(row) == max_step]
    return max_step, float(final_rows[-1][field])


def scalar_summary_metrics(
    runs: list[dict[str, object]],
    split: str,
    max_auc_step: int = 540,
) -> list[dict[str, object]]:
    rows = []

    for run in runs:
        strategy = str(run["strategy"])

        hard_points = split_group_points(
            run,
            split,
            "hard",
            "raw_mse_surprise_mean",
        )

        final_step, final_easy = final_group_value(
            run,
            split,
            "easy",
            "raw_mse_surprise_mean",
        )
        _, final_hard = final_group_value(
            run,
            split,
            "hard",
            "raw_mse_surprise_mean",
        )
        _, final_noise = final_group_value(
            run,
            split,
            "noise",
            "raw_mse_surprise_mean",
        )

        rows.append(
            {
                "strategy": strategy,
                "split": split,
                "auc_until_step": max_auc_step,
                "final_step": final_step,
                "hard_auc_until_step_540": trapezoid_mean(hard_points, max_auc_step),
                "final_easy_raw_mse": final_easy,
                "final_hard_raw_mse": final_hard,
                "final_noise_raw_mse": final_noise,
            }
        )

    uniform = next((row for row in rows if row["strategy"] == "uniform"), None)

    if uniform is not None:
        uniform_auc = float(uniform["hard_auc_until_step_540"])
        uniform_final_hard = float(uniform["final_hard_raw_mse"])

        for row in rows:
            auc = float(row["hard_auc_until_step_540"])
            final_hard = float(row["final_hard_raw_mse"])

            row["hard_auc_improvement_vs_uniform_pct"] = (
                100.0 * (uniform_auc - auc) / uniform_auc
                if uniform_auc != 0.0
                else 0.0
            )
            row["final_hard_improvement_vs_uniform_pct"] = (
                100.0 * (uniform_final_hard - final_hard) / uniform_final_hard
                if uniform_final_hard != 0.0
                else 0.0
            )

    return sorted(
        rows,
        key=lambda row: float(row["hard_auc_until_step_540"]),
    )



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/surprise_sampling_rare_hard.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    root = Path(config["experiment"]["output_root"])
    out = root / "comparison"

    if out.exists():
        import shutil
        shutil.rmtree(out)

    dashboards = out / "dashboards"
    tables = out / "tables"
    dashboards.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    runs = find_latest_runs(root)
    if not runs:
        raise SystemExit("No completed sampling runs with test split found")

    found = {str(run["strategy"]) for run in runs}
    missing = [strategy for strategy in STRATEGIES if strategy not in found]
    if missing:
        print("missing strategies:", ", ".join(missing))

    write_csv(tables / "final_validation_metrics.csv", final_split_metrics(runs, "validation"))
    write_csv(tables / "final_test_metrics.csv", final_split_metrics(runs, "test"))
    write_csv(tables / "final_sampling_metrics.csv", final_sampling_metrics(runs))
    write_csv(tables / "summary_validation_metrics.csv", scalar_summary_metrics(runs, "validation"))
    write_csv(tables / "summary_test_metrics.csv", scalar_summary_metrics(runs, "test"))

    validation_panels = [
        (
            "Easy validation raw MSE",
            make_panel_curves(runs, "summary", "validation", "easy"),
        ),
        (
            "Hard validation raw MSE",
            make_panel_curves(runs, "summary", "validation", "hard"),
        ),
        (
            "Noise validation raw MSE",
            make_panel_curves(runs, "summary", "validation", "noise"),
        ),
    ]

    test_panels = [
        (
            "Easy test raw MSE",
            make_panel_curves(runs, "summary", "test", "easy"),
        ),
        (
            "Hard test raw MSE",
            make_panel_curves(runs, "summary", "test", "hard"),
        ),
        (
            "Noise test raw MSE",
            make_panel_curves(runs, "summary", "test", "noise"),
        ),
    ]

    sampling_panels = [
        (
            "Selected easy fraction",
            make_panel_curves(runs, "sampling", None, "easy"),
        ),
        (
            "Selected hard fraction",
            make_panel_curves(runs, "sampling", None, "hard"),
        ),
        (
            "Selected noise fraction",
            make_panel_curves(runs, "sampling", None, "noise"),
        ),
    ]

    dashboard_svg(
        dashboards / "validation.svg",
        "Validation raw MSE dashboard",
        "Three groups on one page; lower is better for easy/hard, noise is expected to remain difficult",
        validation_panels,
        "raw MSE",
    )

    dashboard_svg(
        dashboards / "test.svg",
        "Test raw MSE dashboard",
        "Final held-out split; use this for honest comparison after choosing strategies on validation",
        test_panels,
        "raw MSE",
    )

    dashboard_svg(
        dashboards / "sampling.svg",
        "Sampling fractions dashboard",
        "How each strategy changes the composition of training batches",
        sampling_panels,
        "selected fraction",
    )

    print(f"wrote dashboard report to {out}")


if __name__ == "__main__":
    main()
