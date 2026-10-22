"""Configuration A (classifier alone) × B (classifier + profile) on DS2 (ARQUITETURA §11.2)."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy import stats

from cardiac_twin.config import Config
from cardiac_twin.evaluation.replay import ReplayResult, replay_record
from cardiac_twin.evaluation.system import ALARM_LEVELS, aggregate, system_metrics
from cardiac_twin.model.inference import BeatClassifier

PAIRED_KEYS = (
    "fp",
    "fp_spurious",
    "alarms",
    "episodes",
    "alarmed_s",
    "alarmed_v",
    "alarmed_sv",
)
COLOR_A = "#2a78d6"
COLOR_B = "#eb6834"
INK = "#52514e"
GRID = "#e4e3df"


@dataclasses.dataclass(frozen=True)
class Comparison:
    rows: list[dict[str, Any]]
    summary: dict[str, Any]
    replays: dict[str, tuple[ReplayResult, ReplayResult]]


def check_b_subset_of_a(a: ReplayResult, b: ReplayResult) -> None:
    """B only removes alarms of A (Quadro 5). Anything else is a bug."""
    if not np.array_equal(a.beat_sample, b.beat_sample) or not np.array_equal(a.beat_cls, b.beat_cls):
        raise ValueError(f"{a.record_id}: A and B detected or classified different beats")
    extra = np.isin(b.beat_level, ALARM_LEVELS) & ~np.isin(a.beat_level, ALARM_LEVELS)
    if extra.any():
        raise ValueError(f"{a.record_id}: {int(extra.sum())} alarms in B that are not in A")


def per_record_row(ma: dict[str, Any], mb: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"record_id": ma["record_id"]}
    row["hours"] = ma["hours"]
    row["n_beats_n"] = ma["n_beats_n"]
    row["total_s"], row["total_v"], row["total_sv"] = ma["total_s"], ma["total_v"], ma["total_sv"]
    for key in PAIRED_KEYS:
        row[f"{key}_a"], row[f"{key}_b"] = ma[key], mb[key]
    for cfg_name in ("a", "b"):
        row[f"fp_per_hour_{cfg_name}"] = row[f"fp_{cfg_name}"] / row["hours"] if row["hours"] else np.nan
    return row


def wilcoxon_fp(fp_a: Sequence[int], fp_b: Sequence[int], alpha: float) -> dict[str, Any]:
    """Paired one-sided Wilcoxon signed-rank test, H1: FP(B) < FP(A) (A-03).

    Zero differences are dropped (zero_method="wilcox"). W is the sum of the ranks of the positive
    differences B − A; scipy selects the exact or the normal distribution.
    """
    a = np.asarray(fp_a, dtype=float)
    b = np.asarray(fp_b, dtype=float)
    diff = b - a
    out: dict[str, Any] = {
        "hypothesis": "FP(B) < FP(A)",
        "n_pairs": int(len(diff)),
        "n_nonzero": int(np.count_nonzero(diff)),
        "median_diff_b_minus_a": float(np.median(diff)),
        "alpha": alpha,
        "scipy": scipy.__version__,
    }
    if out["n_nonzero"] == 0:
        return {**out, "w": None, "p_value": None, "significant": False}
    res = stats.wilcoxon(b, a, zero_method="wilcox", alternative="less")
    return {
        **out,
        "w": float(res.statistic),
        "p_value": float(res.pvalue),
        "significant": bool(res.pvalue < alpha),
    }


def example_records(rows: Sequence[dict[str, Any]]) -> list[str]:
    """Two records for the KL × L figure: the largest FP reduction and the largest loss of S+V alarms."""
    gain = max(rows, key=lambda r: (r["fp_a"] - r["fp_b"], r["record_id"]))
    rest = [r for r in rows if r is not gain]
    cost = max(rest, key=lambda r: (r["alarmed_sv_a"] - r["alarmed_sv_b"], r["record_id"]))
    return [gain["record_id"], cost["record_id"]]


def compare_system(
    cfg: Config, classifier: BeatClassifier, records: Sequence[str], raw_dir: Path
) -> Comparison:
    """Replay each record with A and with B; feedback off (rule 5), calibration excluded (rule 6)."""
    skip = cfg.profile.calibration_seconds
    classes = cfg.data.classes
    source = cfg.engine.rpeak_source
    replays = {}
    metrics_a, metrics_b, rows = [], [], []
    for rid in records:
        a = replay_record(cfg, classifier, rid, raw_dir, use_profile=False, rpeak_source=source)
        b = replay_record(cfg, classifier, rid, raw_dir, use_profile=True, rpeak_source=source)
        check_b_subset_of_a(a, b)
        replays[rid] = (a, b)
        ma, mb = system_metrics(a, classes, skip), system_metrics(b, classes, skip)
        metrics_a.append(ma)
        metrics_b.append(mb)
        rows.append(per_record_row(ma, mb))

    agg_a, agg_b = aggregate(metrics_a), aggregate(metrics_b)
    summary = {
        "records": list(records),
        "skip_seconds": skip,
        "k": cfg.profile.k,
        "window_seconds": cfg.profile.window_seconds,
        "rpeak_source": source,
        "feedback_enabled": False,
        "a": agg_a,
        "b": agg_b,
        "fp_reduction": 1 - agg_b["fp"] / agg_a["fp"] if agg_a["fp"] else float("nan"),
        "sensitivity_drop_pp": {
            name: 100 * (agg_a[f"sensitivity_{name}"] - agg_b[f"sensitivity_{name}"])
            for name in ("s", "v", "sv")
        },
        "chunk_ms": {
            name: {
                "p50": float(np.percentile(ms, 50)),
                "p95": float(np.percentile(ms, 95)),
                "max": float(ms.max()),
            }
            for name, ms in chunk_ms(replays).items()
        },
        "latency_budget_ms": cfg.evaluation.latency_budget_ms,
        "wilcoxon_fp": wilcoxon_fp(
            [r["fp_a"] for r in rows], [r["fp_b"] for r in rows], cfg.evaluation.wilcoxon_alpha
        ),
        "records_fp_b_below_a": sum(r["fp_b"] < r["fp_a"] for r in rows),
        "records_fp_equal": sum(r["fp_b"] == r["fp_a"] for r in rows),
    }
    return Comparison(rows=rows, summary=summary, replays=replays)


def chunk_ms(replays: dict[str, tuple[ReplayResult, ReplayResult]]) -> dict[str, np.ndarray]:
    """Processing time of every chunk, in ms, for A and for B."""
    return {
        name: 1000 * np.concatenate([pair[i].push_seconds for pair in replays.values()])
        for i, name in enumerate(("a", "b"))
    }


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK)
    ax.tick_params(colors=INK, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_fp_per_record(rows: Sequence[dict[str, Any]], path: Path) -> None:
    """FP per record for A and B, records sorted by FP of A."""
    plt = _pyplot()
    rows = sorted(rows, key=lambda r: (-r["fp_a"], r["record_id"]))
    x = np.arange(len(rows))
    width = 0.4
    fig, ax = plt.subplots(figsize=(7.5, 3.4), dpi=150)
    ax.bar(
        x - width / 2,
        [r["fp_a"] for r in rows],
        width - 0.04,
        color=COLOR_A,
        label="A: classificador isolado",
    )
    ax.bar(
        x + width / 2,
        [r["fp_b"] for r in rows],
        width - 0.04,
        color=COLOR_B,
        label="B: classificador + perfil",
    )
    ax.set_yscale("symlog", linthresh=10)
    ax.set_xticks(x, [r["record_id"] for r in rows], rotation=90)
    ax.set_xlabel("Registro (DS2)", color=INK, fontsize=9)
    ax.set_ylabel("Falsos positivos", color=INK, fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_kl_threshold(
    a: ReplayResult, b: ReplayResult, n_index: int, skip_seconds: float, path: Path
) -> None:
    """KL of each window and threshold L over time; below, the alarms of A and B on N and on non-N beats."""
    plt = _pyplot()
    fig, (ax, rug) = plt.subplots(
        2, 1, figsize=(7.5, 3.6), dpi=150, sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    minutes = b.window_t / 60
    ax.plot(minutes, b.window_kl, color=COLOR_B, linewidth=1.2, label="Divergência KL da janela")
    ax.plot(minutes, b.window_threshold, color=INK, linewidth=1.2, linestyle="--", label="Limiar L")
    ax.axvspan(0, skip_seconds / 60, color=GRID, alpha=0.6, linewidth=0)
    ax.text(
        skip_seconds / 120,
        0.95,
        "calibração",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=7,
        color=INK,
    )
    ax.set_ylabel("KL", color=INK, fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    _style(ax)

    rows = []
    for y, (name, r) in enumerate((("A", a), ("B", b))):
        alarm = np.isin(r.beat_level, ALARM_LEVELS) & (r.beat_t > skip_seconds)
        is_n = r.beat_true == n_index
        rows.append(name)
        rug.scatter(
            r.beat_t[alarm & is_n] / 60,
            np.full((alarm & is_n).sum(), y),
            marker="|",
            s=60,
            color=INK,
            linewidths=0.8,
        )
        rug.scatter(
            r.beat_t[alarm & ~is_n] / 60,
            np.full((alarm & ~is_n).sum(), y),
            marker="|",
            s=60,
            color=COLOR_B if name == "B" else COLOR_A,
            linewidths=0.8,
        )
    rug.set_yticks(range(len(rows)), [f"Alertas {n}" for n in rows])
    rug.set_ylim(-0.6, len(rows) - 0.4)
    rug.set_xlabel("Tempo (min)", color=INK, fontsize=9)
    _style(rug)
    rug.grid(False)
    ax.set_title(
        f"Registro {b.record_id}. Cinza: alerta em batimento N; cor: alerta em batimento não N.\n"
        "Alertas críticos (3 ou mais V seguidos) não dependem da KL.",
        fontsize=8,
        color=INK,
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_latency(chunk_ms: dict[str, np.ndarray], budget_ms: float, path: Path) -> None:
    """Processing time of each 1 s chunk, A and B (RNF01)."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(4.2, 3.2), dpi=150)
    data = [chunk_ms["a"], chunk_ms["b"]]
    box = ax.boxplot(
        data,
        widths=0.5,
        patch_artist=True,
        showfliers=True,
        flierprops={"marker": ".", "markersize": 2, "markeredgecolor": INK},
    )
    for patch, color in zip(box["boxes"], (COLOR_A, COLOR_B), strict=True):
        patch.set_facecolor(color)
        patch.set_edgecolor(INK)
    for median in box["medians"]:
        median.set_color("#0b0b0b")
    ax.set_xticks([1, 2], ["A", "B"])
    ax.set_yscale("log")
    ax.set_ylabel("Tempo por bloco de 1 s (ms)", color=INK, fontsize=9)
    ax.set_title(f"Limite do RNF01: {budget_ms:.0f} ms", fontsize=8, color=INK, loc="left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
