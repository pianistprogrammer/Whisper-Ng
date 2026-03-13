#!/usr/bin/env python3
"""
Generate a visual dashboard PNG from test_results_yor.json.

Usage:
    python generate_test_dashboard.py
    python generate_test_dashboard.py --json multilingual_whisper_hf/checkpoint-1000-mlx/test_results_yor.json
    python generate_test_dashboard.py --out dashboard.png
"""

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np


# ── colour palettes ─────────────────────────────────────────────────────────
C_PERFECT  = "#2ecc71"   # green   WER == 0
C_GOOD     = "#27ae60"   # darker  WER < 0.30
C_OK       = "#f39c12"   # amber   0.30 ≤ WER < 0.70
C_BAD      = "#e74c3c"   # red     0.70 ≤ WER < 1.00
C_TERRIBLE = "#8e1c1c"   # dark    WER ≥ 1.00

# dark (default)
_DARK = dict(
    BG="#1a1a2e", PANEL="#16213e", TEXT="#e0e0e0",
    ACCENT="#0f3460", CYAN="#00d4ff", TITLE="#ffffff",
    SUBTITLE="#aaaacc", SPINE="#444466", DIVIDER="#334466",
    BOX_FACE="#0f3460", SCATTER_LINE=C_BAD,
)
# light
_LIGHT = dict(
    BG="#ffffff", PANEL="#f4f6fb", TEXT="#1a1a2e",
    ACCENT="#2c3e7a", CYAN="#1565c0", TITLE="#1a1a2e",
    SUBTITLE="#555577", SPINE="#c0c8d8", DIVIDER="#aab0c8",
    BOX_FACE="#d6e4f7", SCATTER_LINE="#c0392b",
)

# active palette (mutated by apply_theme)
C_BG = C_PANEL = C_TEXT = C_ACCENT = C_CYAN = C_TITLE = ""
C_SUBTITLE = C_SPINE = C_DIVIDER = C_BOX_FACE = C_SCATTER_LINE = ""

def apply_theme(name: str = "dark") -> None:
    """Copy the chosen palette into the module-level colour globals."""
    global C_BG, C_PANEL, C_TEXT, C_ACCENT, C_CYAN, C_TITLE
    global C_SUBTITLE, C_SPINE, C_DIVIDER, C_BOX_FACE, C_SCATTER_LINE
    p = _LIGHT if name == "light" else _DARK
    C_BG=p["BG"]; C_PANEL=p["PANEL"]; C_TEXT=p["TEXT"]
    C_ACCENT=p["ACCENT"]; C_CYAN=p["CYAN"]; C_TITLE=p["TITLE"]
    C_SUBTITLE=p["SUBTITLE"]; C_SPINE=p["SPINE"]; C_DIVIDER=p["DIVIDER"]
    C_BOX_FACE=p["BOX_FACE"]; C_SCATTER_LINE=p["SCATTER_LINE"]

apply_theme("dark")  # default — keeps existing behaviour


def wer_color(w: float) -> str:
    if w == 0:        return C_PERFECT
    if w < 0.30:      return C_GOOD
    if w < 0.70:      return C_OK
    if w < 1.00:      return C_BAD
    return C_TERRIBLE


def wer_label(w: float) -> str:
    if w == 0:        return "Perfect"
    if w < 0.30:      return "Good (<30%)"
    if w < 0.70:      return "Fair (30–70%)"
    if w < 1.00:      return "Poor (70–100%)"
    return "Failed (≥100%)"


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── helpers ─────────────────────────────────────────────────────────────────

def add_panel_bg(ax, color=None, alpha=0.6):
    ax.set_facecolor(color if color is not None else C_PANEL)
    for sp in ax.spines.values():
        sp.set_color(C_SPINE)
        sp.set_linewidth(0.8)


def style_ax(ax, title, xlabel="", ylabel=""):
    add_panel_bg(ax)
    ax.set_title(title, color=C_TEXT, fontsize=9, fontweight="bold", pad=6)
    ax.set_xlabel(xlabel, color=C_SUBTITLE, fontsize=8)
    ax.set_ylabel(ylabel, color=C_SUBTITLE, fontsize=8)
    ax.tick_params(colors=C_TEXT, labelsize=7.5)
    ax.xaxis.label.set_color(C_SUBTITLE)
    ax.yaxis.label.set_color(C_SUBTITLE)
    for tl in ax.get_xticklabels() + ax.get_yticklabels():
        tl.set_color(C_TEXT)


# ── individual plot functions ────────────────────────────────────────────────

def plot_kpi_strip(ax, data: dict):
    """Top strip of big KPI numbers."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(C_BG)  # KPI strip uses full bg

    n  = data["n_samples"]
    samples = data["samples"]
    wers = [s["wer"] for s in samples]

    kpis = [
        ("Samples",       f"{n}",                      C_CYAN),
        ("Mean WER",      f"{data['mean_wer']*100:.1f}%",  "#e74c3c"),
        ("Median WER",    f"{data['median_wer']*100:.1f}%", "#f39c12"),
        ("Perfect (0%)",  f"{sum(1 for w in wers if w==0)}", C_PERFECT),
        ("Good (<30%)",   f"{sum(1 for w in wers if w<0.30)}", C_GOOD),
        ("Failed (≥100%)",f"{sum(1 for w in wers if w>=1.0)}", C_TERRIBLE),
        ("Speed",         f"{data['elapsed_s']/n:.2f}s/sample", "#9b59b6"),
    ]
    ncols = len(kpis)
    for i, (label, value, color) in enumerate(kpis):
        x = (i + 0.5) / ncols
        ax.text(x, 0.72, value, ha="center", va="center", fontsize=16,
                fontweight="bold", color=color, transform=ax.transAxes)
        ax.text(x, 0.28, label, ha="center", va="center", fontsize=7.5,
                color=C_SUBTITLE, transform=ax.transAxes)

    # divider lines
    for i in range(1, ncols):
        ax.axvline(i / ncols, color=C_DIVIDER, lw=0.8)


def plot_wer_histogram(ax, wers):
    bins = np.linspace(0, max(min(wers + [3.0]), 3.0), 31)
    colors_bins = []
    for b in bins[:-1]:
        colors_bins.append(wer_color(b))

    n_hist, edges, patches = ax.hist(wers, bins=bins, edgecolor="#111133", linewidth=0.4)
    for patch, color in zip(patches, colors_bins):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    ax.axvline(np.mean(wers),   color="#ffffff", lw=1.2, linestyle="--", label=f"Mean  {np.mean(wers):.2f}")
    ax.axvline(np.median(wers), color=C_CYAN,    lw=1.2, linestyle=":",  label=f"Median {np.median(wers):.2f}")
    legend = ax.legend(fontsize=7, facecolor=C_BG, edgecolor=C_SPINE, labelcolor=C_TEXT)
    style_ax(ax, "WER Distribution", "Word Error Rate", "# Samples")


def plot_bucket_bars(ax, wers):
    buckets = [
        ("Perfect\n(=0%)",   sum(1 for w in wers if w == 0),        C_PERFECT),
        ("Good\n(<30%)",     sum(1 for w in wers if 0 < w < 0.30),  C_GOOD),
        ("Fair\n(30–70%)",   sum(1 for w in wers if 0.30 <= w < 0.70), C_OK),
        ("Poor\n(70–100%)",  sum(1 for w in wers if 0.70 <= w < 1.0),  C_BAD),
        ("Failed\n(≥100%)",  sum(1 for w in wers if w >= 1.0),      C_TERRIBLE),
    ]
    labels = [b[0] for b in buckets]
    counts = [b[1] for b in buckets]
    colors = [b[2] for b in buckets]
    bars = ax.bar(labels, counts, color=colors, edgecolor="#111133", linewidth=0.5)
    for bar, cnt in zip(bars, counts):
        pct = 100 * cnt / len(wers)
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                f"{cnt}\n({pct:.0f}%)", ha="center", va="bottom", fontsize=7.5, color=C_TEXT)
    style_ax(ax, "Quality Buckets", "", "# Samples")
    ax.set_ylim(0, max(counts) * 1.28)


def plot_wer_scatter(ax, wers):
    x = np.arange(len(wers))
    colors = [wer_color(w) for w in wers]
    ax.scatter(x, wers, c=colors, s=14, alpha=0.85, zorder=3)
    # running mean
    window = 10
    running = np.convolve(wers, np.ones(window)/window, mode="valid")
    ax.plot(np.arange(window-1, len(wers)), running, color=C_CYAN, lw=1.2,
            alpha=0.9, label=f"{window}-sample rolling mean")
    ax.axhline(np.mean(wers),   color="#ffffff", lw=0.8, linestyle="--", alpha=0.6)
    ax.axhline(np.median(wers), color="#f39c12", lw=0.8, linestyle=":",  alpha=0.6)
    ax.axhline(1.0,             color=C_TERRIBLE, lw=0.6, linestyle=":",  alpha=0.4)

    legend = ax.legend(fontsize=7, facecolor=C_BG, edgecolor=C_SPINE, labelcolor=C_TEXT)
    style_ax(ax, "WER per Sample (in order)", "Sample #", "WER")
    ax.set_ylim(-0.1, min(max(wers) * 1.1, 5.0))


def plot_cumulative(ax, wers):
    sorted_w = np.sort(wers)
    cum = np.arange(1, len(sorted_w)+1) / len(sorted_w) * 100
    ax.plot(sorted_w, cum, color=C_CYAN, lw=1.5)
    ax.fill_between(sorted_w, cum, alpha=0.12, color=C_CYAN)
    for threshold, color, label in [(0.3, C_GOOD, "30%"), (0.7, C_OK, "70%"), (1.0, C_BAD, "100%")]:
        pct = np.interp(threshold, sorted_w, cum)
        ax.axvline(threshold, color=color, lw=0.8, linestyle="--", alpha=0.7)
        ax.text(threshold + 0.02, pct - 6, f"{pct:.0f}% ≤ {label}",
                color=color, fontsize=6.5, va="top")
    style_ax(ax, "Cumulative WER Distribution", "WER threshold", "% samples below threshold")
    ax.set_xlim(0, min(max(wers)*1.02, 3.0))
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())


def plot_ref_len_vs_wer(ax, samples):
    ref_lens = [len(s["reference"].split()) for s in samples]
    wers     = [s["wer"] for s in samples]
    colors   = [wer_color(w) for w in wers]
    ax.scatter(ref_lens, wers, c=colors, s=14, alpha=0.75, zorder=3)
    # linear trend
    if len(ref_lens) > 5:
        z = np.polyfit(ref_lens, wers, 1)
        p = np.poly1d(z)
        xs = np.linspace(min(ref_lens), max(ref_lens), 100)
        ax.plot(xs, p(xs), color="#ffffff", lw=0.8, linestyle="--", alpha=0.5)
    style_ax(ax, "Reference Length vs WER", "# words in reference", "WER")
    ax.set_ylim(-0.1, min(max(wers)*1.1, 4.0))


def plot_sample_table(ax, samples, max_rows=25):
    """A styled table of the worst + best samples."""
    ax.axis("off")
    ax.set_facecolor(C_BG)

    # Pick top-10 worst and top-15 best
    sorted_by_wer = sorted(samples, key=lambda s: s["wer"])
    best   = sorted_by_wer[:10]
    worst  = sorted_by_wer[-10:][::-1]
    display = worst + best

    col_labels = ["ID", "WER", "Reference (truncated)", "Hypothesis (truncated)"]
    col_w = [0.09, 0.06, 0.42, 0.43]
    row_h = 1.0 / (len(display) + 2)
    header_y = 1.0 - row_h * 0.6

    # Header
    x_cursor = 0.0
    for label, w in zip(col_labels, col_w):
        ax.text(x_cursor + w/2, header_y, label, ha="center", va="center",
                fontsize=7, fontweight="bold", color=C_TITLE,
                transform=ax.transAxes)
        x_cursor += w

    ax.plot([0, 1], [1.0 - row_h, 1.0 - row_h], color="#445566", lw=0.8, transform=ax.transAxes, clip_on=False)

    for row_i, s in enumerate(display):
        y = 1.0 - row_h * (row_i + 2.0)
        bg_alpha = 0.22 if row_i % 2 == 0 else 0.08
        bg_color = "#c0392b" if s["wer"] >= 1.0 else ("#2ecc71" if s["wer"] < 0.30 else C_ACCENT)
        rect = FancyBboxPatch((0, y - row_h * 0.4), 1, row_h * 0.88,
                               boxstyle="round,pad=0.002", linewidth=0,
                               facecolor=bg_color, alpha=bg_alpha,
                               transform=ax.transAxes, zorder=0)
        ax.add_patch(rect)

        row_data = [
            s["id"],
            f"{s['wer']:.2f}",
            textwrap.shorten(s["reference"],  width=52, placeholder="…"),
            textwrap.shorten(s["hypothesis"], width=52, placeholder="…"),
        ]
        x_cursor = 0.0
        for val, w in zip(row_data, col_w):
            c = wer_color(s["wer"]) if val == row_data[1] else C_TEXT
            ax.text(x_cursor + w/2, y, val, ha="center", va="center",
                    fontsize=6.2, color=c, transform=ax.transAxes)
            x_cursor += w

    # legend separator header
    sep_y = 1.0 - row_h * (10 + 1.5)
    ax.plot([0, 1], [sep_y, sep_y], color=C_DIVIDER, lw=1.0, linestyle="--", transform=ax.transAxes, clip_on=False)
    ax.text(0.5, sep_y + row_h * 0.15, "▲ WORST 10  |  ▼ BEST 10",
            ha="center", va="bottom", fontsize=6.5, color=C_SUBTITLE,
            transform=ax.transAxes)

    ax.set_title("Sample-Level Breakdown (Worst 10 + Best 10)", color=C_TEXT,
                 fontsize=9, fontweight="bold")


def plot_pie(ax, wers):
    buckets = [
        ("Perfect (=0%)",   sum(1 for w in wers if w == 0)),
        ("Good (<30%)",     sum(1 for w in wers if 0 < w < 0.30)),
        ("Fair (30–70%)",   sum(1 for w in wers if 0.30 <= w < 0.70)),
        ("Poor (70–100%)",  sum(1 for w in wers if 0.70 <= w < 1.0)),
        ("Failed (≥100%)",  sum(1 for w in wers if w >= 1.0)),
    ]
    colors = [C_PERFECT, C_GOOD, C_OK, C_BAD, C_TERRIBLE]
    labels = [b[0] for b in buckets]
    sizes  = [b[1] for b in buckets]
    explode = [0.04] * len(sizes)

    wedges, texts, autotexts = ax.pie(
        sizes, labels=None, colors=colors, explode=explode,
        autopct=lambda p: f"{p:.0f}%" if p > 3 else "",
        startangle=140, pctdistance=0.78,
        wedgeprops={"edgecolor": C_BG, "linewidth": 1.2}
    )
    for at in autotexts:
        at.set_color(C_TITLE)
        at.set_fontsize(7.5)
        at.set_fontweight("bold")

    patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
    ax.legend(handles=patches, loc="lower left", fontsize=6.5,
              facecolor=C_PANEL, edgecolor=C_SPINE, labelcolor=C_TEXT,
              framealpha=0.9, bbox_to_anchor=(-0.1, -0.05))
    ax.set_title("Quality Distribution", color=C_TEXT, fontsize=9, fontweight="bold")
    ax.set_facecolor(C_BG)


# ── main layout ──────────────────────────────────────────────────────────────

def build_dashboard(data: dict, out_path: str, theme: str = "dark"):
    apply_theme(theme)
    samples = data["samples"]
    wers    = [s["wer"] for s in samples]

    fig = plt.figure(figsize=(22, 26), facecolor=C_BG)
    fig.patch.set_facecolor(C_BG)

    gs = gridspec.GridSpec(
        5, 3,
        figure=fig,
        hspace=0.52,
        wspace=0.32,
        top=0.955, bottom=0.03,
        left=0.04, right=0.98,
        height_ratios=[0.12, 0.28, 0.28, 0.28, 0.58],
    )

    # ── Row 0: KPI strip ────────────────────────────────────────────────────
    ax_kpi = fig.add_subplot(gs[0, :])
    plot_kpi_strip(ax_kpi, data)

    # ── Row 1 ────────────────────────────────────────────────────────────────
    ax_hist  = fig.add_subplot(gs[1, 0])
    ax_pie   = fig.add_subplot(gs[1, 1])
    ax_buck  = fig.add_subplot(gs[1, 2])

    plot_wer_histogram(ax_hist, wers)
    plot_pie(ax_pie, wers)
    plot_bucket_bars(ax_buck, wers)

    # ── Row 2 ────────────────────────────────────────────────────────────────
    ax_scatter = fig.add_subplot(gs[2, :2])
    ax_cum     = fig.add_subplot(gs[2, 2])

    plot_wer_scatter(ax_scatter, wers)
    plot_cumulative(ax_cum, wers)

    # ── Row 3 ────────────────────────────────────────────────────────────────
    ax_len   = fig.add_subplot(gs[3, 0])
    plot_ref_len_vs_wer(ax_len, samples)

    # Stats box
    ax_stats = fig.add_subplot(gs[3, 1])
    ax_stats.axis("off")
    ax_stats.set_facecolor(C_PANEL)
    add_panel_bg(ax_stats)
    p25, p50, p75, p90, p95 = np.percentile(wers, [25, 50, 75, 90, 95])
    stats_lines = [
        ("Model",          data["model"].split("/")[-1]),
        ("Dataset",        data["dataset"]),
        ("",               ""),
        ("Samples",        f"{data['n_samples']}"),
        ("Mean WER",       f"{data['mean_wer']*100:.1f}%"),
        ("Median WER",     f"{data['median_wer']*100:.1f}%"),
        ("Std Dev",        f"{np.std(wers)*100:.1f}%"),
        ("",               ""),
        ("P25",            f"{p25*100:.1f}%"),
        ("P50 (median)",   f"{p50*100:.1f}%"),
        ("P75",            f"{p75*100:.1f}%"),
        ("P90",            f"{p90*100:.1f}%"),
        ("P95",            f"{p95*100:.1f}%"),
        ("",               ""),
        ("Speed",          f"{data['elapsed_s']/data['n_samples']:.2f} s/sample"),
        ("Total time",     f"{data['elapsed_s']:.1f}s"),
    ]
    y_start = 0.96
    for label, value in stats_lines:
        if label == "" and value == "":
            y_start -= 0.028
            continue
        ax_stats.text(0.08, y_start, label + ":", transform=ax_stats.transAxes,
                      fontsize=7.5, color=C_SUBTITLE, va="top")
        ax_stats.text(0.55, y_start, value, transform=ax_stats.transAxes,
                      fontsize=7.5, color=C_TEXT, va="top", fontweight="bold")
        y_start -= 0.055
    ax_stats.set_title("Summary Statistics", color=C_TEXT, fontsize=9, fontweight="bold")

    # WER box-plot
    ax_box = fig.add_subplot(gs[3, 2])
    bp = ax_box.boxplot(
        [wers], vert=True, patch_artist=True, widths=0.4,
        flierprops=dict(marker="o", markersize=3, markerfacecolor=C_BAD, alpha=0.5),
        medianprops=dict(color=C_CYAN, lw=2),
        boxprops=dict(facecolor=C_BOX_FACE, linewidth=0.8),
        whiskerprops=dict(color=C_TEXT, linewidth=0.8),
        capprops=dict(color=C_TEXT, linewidth=0.8),
    )
    style_ax(ax_box, "WER Box Plot", "", "WER")
    ax_box.set_xticks([])
    ax_box.set_xlim(0.5, 1.5)

    # ── Row 4: sample table ──────────────────────────────────────────────────
    ax_table = fig.add_subplot(gs[4, :])
    plot_sample_table(ax_table, samples)

    # ── Main title ──────────────────────────────────────────────────────────
    fig.text(
        0.5, 0.976,
        "Whisper Fine-Tune Evaluation  ·  yor_tts Test Set",
        ha="center", va="top", fontsize=16, fontweight="bold",
        color=C_TITLE, family="DejaVu Sans",
    )
    fig.text(
        0.5, 0.962,
        f"Model: {data['model']}   |   Dataset: {data['dataset']}   |   "
        f"{data['n_samples']} samples   |   Mean WER: {data['mean_wer']*100:.1f}%   |   Median WER: {data['median_wer']*100:.1f}%",
        ha="center", va="top", fontsize=8.5, color=C_SUBTITLE,
    )

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=C_BG)
    print(f"✓ Dashboard saved → {out_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate a test-result dashboard PNG")
    parser.add_argument(
        "--json",
        default="multilingual_whisper_hf/checkpoint-1000-mlx/test_results_yor.json",
        help="Path to test_results_yor.json",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output PNG path (default: dashboard_yor.png or dashboard_yor_light.png)",
    )
    parser.add_argument(
        "--theme",
        default="dark",
        choices=["dark", "light"],
        help="Colour theme: dark (default) or light (white background)",
    )
    args = parser.parse_args()

    json_path = Path(args.json)
    suffix    = "_light" if args.theme == "light" else ""
    out_path  = args.out or str(json_path.parent / f"dashboard_yor{suffix}.png")

    data = load(json_path)
    build_dashboard(data, out_path, theme=args.theme)


if __name__ == "__main__":
    main()
