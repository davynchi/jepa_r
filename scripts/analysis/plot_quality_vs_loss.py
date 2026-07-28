#!/usr/bin/env python3
"""Plot checkpoint-level Shapes3D quality metrics against held-out JEPA loss."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from jepa.analysis.quality_metrics import METRIC_SPECS  # noqa: E402


COLORS = {
    "uniform": "#2563EB",
    "loss": "#D97706",
    "ras-logdet": "#7C3AED",
    "coord-covariance": "#0F766E",
    "coord-transformation": "#DB2777",
    "coord-dynamics": "#64748B",
}
MODEL_MARKERS = {
    "uniform": "o",
    "loss": "s",
    "ras-logdet": "^",
    "coord-covariance": "D",
}
MODEL_LABELS = {
    "uniform": "○ uniform model",
    "loss": "□ loss model",
    "ras-logdet": "△ RAS model",
    "coord-covariance": "◇ covariance model",
}
FORMULAS = {
    "q1_cross_covariance": (
        "Межподпространственная ковариация",
        r"$\|C_{\rm off}\|_F/(\|C\|_F+\epsilon)$",
    ),
    "q2_projector_interaction": (
        "Взаимодействие проекторов",
        r"$\sqrt{\sum_{i\ne j}\|P_iCP_j\|_F^2}/(\|C\|_F+\epsilon)$",
    ),
    "q3_projector_overlap": (
        "Перекрытие подпространств",
        r"$\sqrt{\sum_{i\ne j}\|U_i^\top U_j\|_F^2}$",
    ),
    "q4_partition_incompleteness": (
        "Неполнота разбиения",
        r"$\|I-\sum_iP_i\|_F/\sqrt{d}$",
    ),
    "q5_subspace_similarity": (
        "Сходство подпространств",
        r"$\sum_i\frac{r_i}{d}\frac{\|U_i^{t\top}U_i^{t-1}\|_F^2}{r_i}$",
    ),
    "q6_subspace_distance": (
        "Расстояние между подпространствами",
        r"$\sqrt{\sum_i\frac{r_i}{d}\frac{\|P_i^t-P_i^{t-1}\|_F^2}{2r_i}}$",
    ),
    "q7_cross_jacobian_energy": (
        "Межподпространственная энергия Jacobian",
        r"$\sqrt{\sum_{i\ne j}\|P_iJP_j\|_F^2/\|J\|_F^2}$",
    ),
    "q8_virtual_sensitivity_magnitude": (
        "Чувствительность к виртуальному update",
        r"$\mathbb{E}_x\sum_i\delta_i(x)$",
    ),
    "q9_subspace_velocity": (
        "Скорость подпространств",
        r"$Q_6/(step_t-step_{t-1})$",
    ),
    "q10_weighted_shape_entropy": (
        "Взвешенная shape-entropy",
        r"$\sum_i\frac{r_i}{d}\frac{\log\det(r_i\Sigma_i/tr\Sigma_i+\epsilon I)}{2r_i}$",
    ),
    "q11_cross_subspace_gaussian_mi": (
        "Gaussian MI между подпространствами",
        r"$\operatorname{avg}_{i<j} I_G(V_i;V_j)/\min(r_i,r_j)$",
    ),
    "q12_absolute_entropy_change": (
        "Изменение entropy",
        r"$|Q_{10}^{t}-Q_{10}^{t-1}|$",
    ),
    "q13_realized_surprise_locality": (
        "Локальность surprise",
        r"$1-H(p_\delta)/\log K,\quad p_{\delta,i}=G_i/\sum_jG_j$",
    ),
    "q14_gradient_locality": (
        "Локальность градиента",
        r"$1-H(e)/\log K,\quad e_i=\|\nabla L\,U_i\|^2/\|\nabla L\|^2$",
    ),
    "q15_virtual_interference": (
        "Интерференция виртуального update",
        r"$\mathbb{E}\|z_{after}-z_{before}\|^2/(\mathbb{E}\|z_{before}\|^2+\epsilon)$",
    ),
    "q16_entity_consistency": (
        "Стабильность entity-проекции",
        r"$\mathbb{E}\|P_Ez-P_Ez'\|^2/(\mathbb{E}\|P_Ez\|^2+\epsilon)$",
    ),
    "q17_mask_transformation_residual": (
        "Остаток линейного преобразования",
        r"$\|\hat Tz_a-z_b\|^2/\|z_b-\bar z_b\|^2$",
    ),
    "q18_mask_perturbation_concentration": (
        "Концентрация perturbation",
        r"$\mathbb{E}[1-H(e(\Delta z))/\log K]$",
    ),
    "q19a_jacobian_effective_rank": (
        "Эффективный ранг Jacobian",
        r"$\exp H(\sigma^2/\sum_j\sigma_j^2)/\min(\dim J)$",
    ),
    "q19b_jacobian_density_tau_1em3": (
        "Плотность Jacobian",
        r"$\operatorname{mean}\,\mathbf{1}[|J_{ab}|>10^{-3}\max|J|]$",
    ),
    "q20_reconstruction_nmse": (
        "Ошибка линейной реконструкции",
        r"$\|\hat Dz-x\|^2/\|x-\bar x\|^2$",
    ),
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-root", type=Path, required=True)
    return parser.parse_args()


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2 + 1
        start = end
    return ranks


def _correlation(x: np.ndarray, y: np.ndarray, *, ranks: bool = False) -> float | None:
    if (
        len(x) < 8
        or np.ptp(x) == 0
        or np.ptp(y) == 0
        or float(np.max(np.abs(x))) <= 1e-12
    ):
        return None
    if ranks:
        x, y = _ranks(x), _ranks(y)
    return float(np.corrcoef(x, y)[0, 1])


def _metric_rows(root: Path, outcome: str) -> dict[str, list[dict]]:
    losses = {
        (row["run_id"], row["checkpoint_id"]): row
        for row in _jsonl(root / "checkpoint_losses.jsonl")
    }
    accuracies = {
        (row["run_id"], row["checkpoint_id"]): row
        for row in _jsonl(root / "accuracy.jsonl")
    }
    output: dict[str, list[dict]] = defaultdict(list)
    for row in _jsonl(root / "records.jsonl"):
        if row["state"] != "complete" or row["value"] is None:
            continue
        expected_panel = (
            "supervised_lda" if row["metric_name"] == "q16_entity_consistency" else "label_free"
        )
        if row["panel"] != expected_panel:
            continue
        loss = losses.get((row["run_id"], row["checkpoint_id"]))
        accuracy = accuracies.get((row["run_id"], row["checkpoint_id"]))
        if loss is None or accuracy is None:
            continue
        diagnostics = json.loads(row["diagnostics_json"])
        output[row["metric_name"]].append(
            {
                "metric": float(row["value"]),
                "outcome": (
                    float(loss["loss"])
                    if outcome == "loss"
                    else float(accuracy["accuracy"])
                ),
                "epoch": int(loss["epoch"]),
                "curriculum": diagnostics["curriculum"],
                "run_id": row["run_id"],
            }
        )
    return output


def _export_values(root: Path, output: Path) -> Path:
    losses = {
        (row["run_id"], row["checkpoint_id"]): row
        for row in _jsonl(root / "checkpoint_losses.jsonl")
    }
    accuracies = {
        (row["run_id"], row["checkpoint_id"]): row
        for row in _jsonl(root / "accuracy.jsonl")
    }
    q_by_name = {spec.name: spec.q_number for spec in METRIC_SPECS}
    rows: list[dict] = []
    for record in _jsonl(root / "records.jsonl"):
        expected_panel = (
            "supervised_lda"
            if record["metric_name"] == "q16_entity_consistency"
            else "label_free"
        )
        if record["panel"] != expected_panel:
            continue
        loss = losses[(record["run_id"], record["checkpoint_id"])]
        accuracy = accuracies[(record["run_id"], record["checkpoint_id"])]
        diagnostics = json.loads(record["diagnostics_json"])
        rows.append(
            {
                "run_id": record["run_id"],
                "curriculum": diagnostics["curriculum"],
                "checkpoint_id": record["checkpoint_id"],
                "epoch": loss["epoch"],
                "q_number": q_by_name[record["metric_name"]],
                "metric_name": record["metric_name"],
                "panel": record["panel"],
                "q_value": record["value"],
                "null_reason": record["null_reason"],
                "heldout_jepa_loss": loss["loss"],
                "classification_accuracy": accuracy["accuracy"],
                "balanced_accuracy": accuracy["balanced_accuracy"],
                "loss_sample_count": loss["sample_count"],
            }
        )
    destination = output / "quality_vs_loss_values.csv"
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return destination


def _style_axis(axis: plt.Axes) -> None:
    axis.grid(color="#E2E8F0", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(colors="#475569", labelsize=8)
    axis.xaxis.label.set_color("#334155")
    axis.yaxis.label.set_color("#334155")
    axis.title.set_color("#0F172A")


def _draw_metric(
    axis: plt.Axes,
    name: str,
    q_number: str,
    rows: list[dict],
    *,
    outcome: str,
) -> dict:
    x = np.asarray([row["metric"] for row in rows], dtype=float)
    y = np.asarray([row["outcome"] for row in rows], dtype=float)
    for row in rows:
        axis.scatter(
            row["metric"],
            row["outcome"],
            color=COLORS.get(row["curriculum"], "#64748B"),
            marker=MODEL_MARKERS.get(row["curriculum"], "o"),
            s=31,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        offsets = {
            25: (4, 4),
            50: (4, -10),
            75: (-13, 4),
            100: (-17, -10),
        }
        axis.annotate(
            str(row["epoch"]),
            (row["metric"], row["outcome"]),
            textcoords="offset points",
            xytext=offsets.get(row["epoch"], (4, 4)),
            fontsize=6.5,
            color=COLORS.get(row["curriculum"], "#475569"),
            zorder=4,
        )
    pearson = _correlation(x, y)
    spearman = _correlation(x, y, ranks=True)
    if pearson is not None:
        coefficients = np.polyfit(x, y, 1)
        grid = np.linspace(float(x.min()), float(x.max()), 100)
        axis.plot(grid, np.polyval(coefficients, grid), color="#64748B", linewidth=1)
    r = "NA" if pearson is None else f"{pearson:+.2f}"
    rho = "NA" if spearman is None else f"{spearman:+.2f}"
    axis.set_title(f"Q{q_number}", loc="left", fontsize=10, fontweight="semibold")
    axis.text(
        0.98,
        0.96,
        f"n={len(rows)}  r={r}  ρ={rho}",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#475569",
    )
    axis.set_xlabel("Q value", fontsize=8)
    axis.set_ylabel(
        "Held-out JEPA loss" if outcome == "loss" else "Classification accuracy",
        fontsize=8,
    )
    if outcome == "loss":
        axis.set_yscale("log")
    axis.margins(x=0.09, y=0.1)
    display_name, formula = FORMULAS[name]
    axis.text(
        0,
        -0.25,
        f"{display_name}\n{formula}",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        color="#334155",
    )
    axis.xaxis.set_major_locator(MaxNLocator(5))
    axis.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3), useOffset=True)
    _style_axis(axis)
    return {
        "metric_name": name,
        "q_number": q_number,
        "n": len(rows),
        "pearson": pearson,
        "spearman": spearman,
    }


def _pages(
    root: Path, rows_by_metric: dict[str, list[dict]], *, outcome: str
) -> list[dict]:
    specs = list(METRIC_SPECS)
    results: list[dict] = []
    for page_index, start in enumerate(range(0, len(specs), 7), start=1):
        page_specs = specs[start : start + 7]
        figure, axes = plt.subplots(2, 4, figsize=(15, 10.2))
        flat = list(axes.flat)
        for axis, spec in zip(flat, page_specs, strict=False):
            results.append(
                _draw_metric(
                    axis,
                    spec.name,
                    spec.q_number,
                    rows_by_metric.get(spec.name, []),
                    outcome=outcome,
                )
            )
        for axis in flat[len(page_specs) :]:
            axis.remove()
        figure.suptitle(
            (
                "Shapes3D quality metrics vs held-out JEPA loss"
                if outcome == "loss"
                else "Shapes3D quality metrics vs classification accuracy"
            )
            + f" · page {page_index}",
            x=0.025,
            y=0.985,
            ha="left",
            fontsize=15,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.025,
            0.945,
            "Point shape and color identify the model. Small labels are checkpoint epochs. "
            "r = Pearson; ρ = Spearman. Correlations are omitted for n<8.",
            fontsize=8,
            color="#64748B",
        )
        legend_handles = [
            Line2D(
                [],
                [],
                color=COLORS[model],
                marker=MODEL_MARKERS[model],
                linestyle="None",
                markersize=6,
                label=MODEL_LABELS[model],
            )
            for model in MODEL_MARKERS
        ]
        figure.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.018),
            ncol=4,
            frameon=False,
            title="Форма точки = модель",
            fontsize=8,
            title_fontsize=8,
        )
        figure.subplots_adjust(
            left=0.07,
            right=0.985,
            top=0.88,
            bottom=0.14,
            hspace=0.72,
            wspace=0.34,
        )
        figure.savefig(
            root / f"quality_vs_{outcome}_page_{page_index}.png",
            dpi=180,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(figure)
    return results


def _summary(root: Path, results: list[dict], *, outcome: str) -> None:
    valid = [
        row
        for row in results
        if row["pearson"] is not None and row["spearman"] is not None
    ]
    valid.sort(
        key=lambda row: max(abs(row["pearson"]), abs(row["spearman"]))
    )
    figure, axis = plt.subplots(figsize=(10, max(5, 0.34 * len(valid))))
    labels = [f"Q{row['q_number']} · {row['metric_name']}" for row in valid]
    positions = np.arange(len(valid))
    pearson_values = [row["pearson"] for row in valid]
    spearman_values = [row["spearman"] for row in valid]
    height = 0.36
    axis.barh(
        positions - height / 2,
        pearson_values,
        height=height,
        color="#2563EB",
        label="Pearson r",
    )
    axis.barh(
        positions + height / 2,
        spearman_values,
        height=height,
        color="#D97706",
        label="Spearman ρ",
    )
    axis.set_yticks(positions, labels)
    axis.axvline(0, color="#0F172A", linewidth=0.8)
    axis.set_xlim(-1, 1)
    axis.set_xlabel(
        "Correlation with held-out JEPA loss"
        if outcome == "loss"
        else "Correlation with classification accuracy"
    )
    axis.set_title(
        "Checkpoint-level Q–loss correlations"
        if outcome == "loss"
        else "Checkpoint-level Q–accuracy correlations",
        loc="left",
        fontweight="bold",
    )
    for index, row in enumerate(valid):
        value = max(
            (float(row["pearson"]), float(row["spearman"])),
            key=abs,
        )
        axis.text(
            value + (0.02 if value >= 0 else -0.02),
            index,
            f"n={row['n']}",
            ha="left" if value >= 0 else "right",
            va="center",
            fontsize=8,
            color="#475569",
        )
    axis.legend(frameon=False, loc="lower right")
    _style_axis(axis)
    figure.tight_layout()
    figure.savefig(
        root / f"quality_vs_{outcome}_correlations.png",
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def _write_interpretation(
    root: Path,
    results_by_outcome: dict[str, list[dict]],
) -> Path:
    def strongest(outcome: str, coefficient: str) -> str:
        valid = [
            row
            for row in results_by_outcome[outcome]
            if row[coefficient] is not None
        ]
        valid.sort(key=lambda row: abs(row[coefficient]), reverse=True)
        lines = []
        for row in valid[:8]:
            display_name = FORMULAS[row["metric_name"]][0]
            symbol = "r" if coefficient == "pearson" else "rho"
            lines.append(
                f"  Q{row['q_number']} ({display_name}): "
                f"{symbol}={row[coefficient]:+.3f}, n={row['n']}"
            )
        return "\n".join(lines)

    selected_records = []
    for row in _jsonl(root.parent / "records.jsonl"):
        expected_panel = (
            "supervised_lda"
            if row["metric_name"] == "q16_entity_consistency"
            else "label_free"
        )
        if row["panel"] == expected_panel:
            selected_records.append(row)
    reason_counts: dict[str, int] = defaultdict(int)
    for row in selected_records:
        reason = "valid" if row["value"] is not None else str(row["null_reason"])
        reason_counts[reason] += 1

    text = f"""ИНТЕРПРЕТАЦИЯ ГРАФИКОВ Q_i ДЛЯ SHAPES3D

1. ЧТО ОЗНАЧАЕТ ТОЧКА

Каждая точка соответствует одной паре:

    (модель, checkpoint)

Исследуются четыре завершённые модели и четыре checkpoint каждой модели:
25, 50, 75 и 100. Поэтому максимально возможно 4 x 4 = 16 точек для одного Q_i.

Форма и цвет точки обозначают модель:

    ○ синяя окружность     — uniform model
    □ оранжевый квадрат    — loss model
    △ фиолетовый треугольник — RAS model
    ◇ зелёный ромб         — covariance model

Маленькое число рядом с точкой — эпоха checkpoint: 25, 50, 75 или 100.

2. ОСИ

Ось X: значение соответствующей метрики Q_i.

Ось Y на графиках loss:
held-out JEPA loss на том же checkpoint. Ниже означает меньший loss.

Ось Y на графиках accuracy:
точность frozen linear classifier на том же checkpoint. Выше означает
лучшую classification accuracy. Encoder заморожен; обучается только линейный probe.

3. ЛИНИЯ И КОРРЕЛЯЦИЯ

Серая линия показывает линейное направление связи между Q_i и результатом.

r — линейная корреляция Пирсона:

    r > 0: наблюдается положительная линейная связь;
    r < 0: наблюдается отрицательная линейная связь;
    r около 0: выраженной линейной связи не видно.

rho — ранговая корреляция Спирмена:

    rho > 0: при увеличении Q_i значение по оси Y обычно увеличивается;
    rho < 0: при увеличении Q_i значение по оси Y обычно уменьшается;
    rho около 0: явной монотонной связи не видно.

Пирсон чувствительнее к выбросам и форме шкалы. Для loss он рассчитан по исходным
значениям loss, хотя для читаемости ось loss на графике показана логарифмически.
Спирмен использует ранги и лучше отражает произвольную монотонную связь.

n — число пригодных точек. Если n < 8, линия, r и rho намеренно не вычисляются:
настолько маленькая выборка слишком легко создаёт случайную сильную корреляцию.

4. ПОЧЕМУ n РАЗНОЕ

Всего возможно 16 точек, но не каждое Q_i определено на каждом checkpoint:

    degenerate_partition: собственные значения совпали на границе подпространств,
    поэтому однозначное разбиение невозможно;

    not_scheduled: дорогие Q7, Q8, Q14, Q15 и Q19 вычислялись только на
    запланированных checkpoints;

    no_previous_checkpoint: Q5, Q6, Q9, Q12 и Q13 требуют предыдущего
    корректного checkpoint.

В выбранных для графиков панелях:

    valid: {reason_counts.get('valid', 0)}
    degenerate_partition: {reason_counts.get('degenerate_partition', 0)}
    not_scheduled: {reason_counts.get('not_scheduled', 0)}
    no_previous_checkpoint: {reason_counts.get('no_previous_checkpoint', 0)}

5. ФОРМУЛЫ

Под каждой панелью написаны русское название Q_i и сокращённая формула,
соответствующая реализации. P_i — ортогональный проектор на V_i,
U_i — ортонормированный базис V_i, C — covariance representation,
J — Jacobian, r_i — размерность V_i, d — полная latent dimension.

6. НАИБОЛЕЕ СИЛЬНЫЕ НАБЛЮДАЕМЫЕ СВЯЗИ С LOSS

По Пирсону:
{strongest('loss', 'pearson')}

По Спирмену:
{strongest('loss', 'spearman')}

Для loss положительная корреляция означает: больше Q_i связано с большим loss.
Отрицательная корреляция означает: больше Q_i связано с меньшим loss.

7. НАИБОЛЕЕ СИЛЬНЫЕ НАБЛЮДАЕМЫЕ СВЯЗИ С ACCURACY

По Пирсону:
{strongest('accuracy', 'pearson')}

По Спирмену:
{strongest('accuracy', 'spearman')}

Для accuracy положительная корреляция означает: больше Q_i связано с большей
accuracy. Отрицательная корреляция означает: больше Q_i связано с меньшей accuracy.

8. ОГРАНИЧЕНИЯ

Эти результаты являются предварительными:

  - checkpoints одной модели не являются независимыми наблюдениями;
  - общий эффект эпохи может одновременно менять Q_i, loss и accuracy;
  - используются только четыре модели и один общий seed block;
  - корреляция не доказывает причинную связь;
  - для финального вывода нужны дополнительные независимые seeds.

Графики loss:
  quality_vs_loss_page_1.png
  quality_vs_loss_page_2.png
  quality_vs_loss_page_3.png

Графики accuracy:
  quality_vs_accuracy_page_1.png
  quality_vs_accuracy_page_2.png
  quality_vs_accuracy_page_3.png

Полные численные данные:
  quality_vs_loss_values.csv
"""
    destination = root / "INTERPRETATION_RU.txt"
    destination.write_text(text)
    return destination


def main() -> None:
    arguments = _args()
    root = arguments.quality_root.expanduser().resolve()
    output = root / "plots"
    output.mkdir(parents=True, exist_ok=True)
    _export_values(root, output)
    results_by_outcome: dict[str, list[dict]] = {}
    for outcome in ("loss", "accuracy"):
        rows_by_metric = _metric_rows(root, outcome)
        results = _pages(output, rows_by_metric, outcome=outcome)
        results_by_outcome[outcome] = results
        _summary(output, results, outcome=outcome)
        (output / f"quality_vs_{outcome}_correlations.json").write_text(
            json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
    _write_interpretation(output, results_by_outcome)
    print(f"metrics={len(METRIC_SPECS)} outcomes=2 output={output}")


if __name__ == "__main__":
    main()
