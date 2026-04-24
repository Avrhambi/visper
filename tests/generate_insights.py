"""Generate insights report + graphs from local config test results."""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT_DIR = Path(__file__).parent
PLOT_FILE = OUT_DIR / "local_config_insights.png"

# ── Data ──────────────────────────────────────────────────────────────────────

BUCKETS = ["short\n(5s)", "medium\n(20s)", "long\n(45s)", "extended\n(90s)"]
BUDGET  = 0.85

rtf_data = {
    "CUDA int8_float32\n(MX350)":      [0.55,  0.18,  0.16,  0.13],
    "iGPU OpenVINO\n(Iris Xe)":        [0.788, 0.259, 0.233, 0.233],
    "HETERO iGPU+CPU\n(OpenVINO)":     [0.776, 0.241, 0.233, 0.226],
    "OV CPU\n(OpenVINO)":              [1.404, 0.390, 0.368, 0.389],
    "CT2 CPU int8\n(best: t2w1)":      [2.57,  0.71,  0.63,  0.50],
}

colors = {
    "CUDA int8_float32\n(MX350)":      "#2ecc71",
    "iGPU OpenVINO\n(Iris Xe)":        "#3498db",
    "HETERO iGPU+CPU\n(OpenVINO)":     "#9b59b6",
    "OV CPU\n(OpenVINO)":              "#e67e22",
    "CT2 CPU int8\n(best: t2w1)":      "#e74c3c",
}

wer_data = {
    "CUDA int8_float32\nVAD on":  {"wer": 0.179, "cer": 0.084, "halluc": 2},
    "CUDA int8_float32\nVAD off": {"wer": 0.180, "cer": 0.084, "halluc": 4},
}

streaming_data = {
    "ct2": {"median": 2719, "p95": 2875},
    "ct4": {"median": 2799, "p95": 2918},
    "ct6": {"median": 2744, "p95": 2854},
}

cold_start = {
    "CUDA\nint8_float32": 5.6,
    "iGPU\n(GPU.0)":      22.1,
    "HETERO\niGPU+CPU":   18.5,
    "OV CPU":              1.9,
    "CT2 CPU\nint8":       4.9,
}

# Colab sweep accuracy comparison
sweep_accuracy = {
    "Default\nfloat16":          {"wer": 0.150, "cer": 0.089, "halluc": 2},
    "Default\nint8_float16":     {"wer": 0.151, "cer": 0.091, "halluc": 1},
    "Baseline\nint8_float16":    {"wer": 0.147, "cer": 0.085, "halluc": 0},
    "Local test\nCUDA int8_f32": {"wer": 0.179, "cer": 0.084, "halluc": 2},
}

# ── Figure layout ─────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(20, 18))
fig.patch.set_facecolor("#1a1a2e")
title_kw = dict(color="white", fontsize=13, fontweight="bold", pad=10)
ax_kw    = dict(facecolor="#16213e")

def style_ax(ax):
    ax.set_facecolor("#16213e")
    ax.tick_params(colors="white", labelsize=9)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35,
                      left=0.06, right=0.97, top=0.93, bottom=0.05)

fig.suptitle("Hebrew STT — Local Hardware Benchmark  |  ivrit-ai/whisper-large-v3-turbo-ct2",
             color="white", fontsize=15, fontweight="bold", y=0.97)

# ── Plot 1: RTF by device across buckets (grouped bar) ────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
style_ax(ax1)

x      = np.arange(4)
n      = len(rtf_data)
width  = 0.15
offset = np.linspace(-(n-1)/2, (n-1)/2, n) * width

for i, (label, vals) in enumerate(rtf_data.items()):
    bars = ax1.bar(x + offset[i], vals, width, label=label,
                   color=colors[label], alpha=0.9, zorder=3)
    for bar, val in zip(bars, vals):
        if val > BUDGET:
            bar.set_edgecolor("#ff0000")
            bar.set_linewidth(2)

ax1.axhline(BUDGET, color="#ff6b6b", linestyle="--", linewidth=1.5,
            label=f"RTF budget ({BUDGET})", zorder=4)
ax1.set_xticks(x)
ax1.set_xticklabels(BUCKETS, color="white")
ax1.set_ylabel("RTF (lower = faster)", color="white")
ax1.set_title("RTF per Bucket — All Devices", **title_kw)
ax1.legend(fontsize=7.5, loc="upper right", facecolor="#0f3460", labelcolor="white",
           framealpha=0.9, ncol=2)
ax1.set_ylim(0, 3.0)
ax1.grid(axis="y", color="#333", linewidth=0.5, zorder=0)

# ── Plot 2: Cold-start time ────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
style_ax(ax2)

cs_labels = list(cold_start.keys())
cs_vals   = list(cold_start.values())
cs_colors = ["#2ecc71", "#3498db", "#9b59b6", "#e67e22", "#e74c3c"]
bars = ax2.barh(cs_labels, cs_vals, color=cs_colors, alpha=0.9)
for bar, val in zip(bars, cs_vals):
    ax2.text(val + 0.2, bar.get_y() + bar.get_height()/2,
             f"{val:.1f}s", va="center", color="white", fontsize=8)
ax2.set_xlabel("Cold-start (seconds)", color="white")
ax2.set_title("Model Load Time", **title_kw)
ax2.grid(axis="x", color="#333", linewidth=0.5)
ax2.tick_params(labelsize=8)

# ── Plot 3: Medium/Long/Extended RTF comparison (best per device) ─────────────
ax3 = fig.add_subplot(gs[1, 0])
style_ax(ax3)

devices_short = ["CUDA\nint8_f32", "iGPU\nOV", "HETERO\nOV", "OV CPU", "CT2 CPU\nint8"]
rtf_medium    = [0.18, 0.259, 0.241, 0.390, 0.71]
bar_colors    = ["#2ecc71", "#3498db", "#9b59b6", "#e67e22", "#e74c3c"]
bars = ax3.bar(devices_short, rtf_medium, color=bar_colors, alpha=0.9)
ax3.axhline(BUDGET, color="#ff6b6b", linestyle="--", linewidth=1.5)
for bar, val in zip(bars, rtf_medium):
    ax3.text(bar.get_x() + bar.get_width()/2, val + 0.01,
             f"{val:.2f}", ha="center", color="white", fontsize=8)
ax3.set_ylabel("RTF", color="white")
ax3.set_title("Medium Audio (20s) RTF", **title_kw)
ax3.set_ylim(0, 1.0)
ax3.grid(axis="y", color="#333", linewidth=0.5)
ax3.tick_params(labelsize=7.5)

# ── Plot 4: Extended RTF comparison ───────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
style_ax(ax4)

rtf_extended = [0.13, 0.233, 0.226, 0.389, 0.50]
bars = ax4.bar(devices_short, rtf_extended, color=bar_colors, alpha=0.9)
ax4.axhline(BUDGET, color="#ff6b6b", linestyle="--", linewidth=1.5)
for bar, val in zip(bars, rtf_extended):
    ax4.text(bar.get_x() + bar.get_width()/2, val + 0.005,
             f"{val:.2f}", ha="center", color="white", fontsize=8)
ax4.set_ylabel("RTF", color="white")
ax4.set_title("Extended Audio (90s) RTF", **title_kw)
ax4.set_ylim(0, 0.7)
ax4.grid(axis="y", color="#333", linewidth=0.5)
ax4.tick_params(labelsize=7.5)

# ── Plot 5: Streaming latency ─────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
style_ax(ax5)

stream_labels  = ["ct2\n(2 threads)", "ct4\n(4 threads)", "ct6\n(6 threads)"]
stream_medians = [2719, 2799, 2744]
stream_p95     = [2875, 2918, 2854]
x_s = np.arange(3)
ax5.bar(x_s - 0.2, stream_medians, 0.35, label="Median", color="#2ecc71", alpha=0.9)
ax5.bar(x_s + 0.2, stream_p95,    0.35, label="p95",    color="#27ae60", alpha=0.7)
ax5.axhline(4000, color="#ff6b6b", linestyle="--", linewidth=1.5, label="4000ms budget")
ax5.set_xticks(x_s)
ax5.set_xticklabels(stream_labels, color="white")
ax5.set_ylabel("Latency (ms)", color="white")
ax5.set_title("Streaming Chunk Latency\n(CUDA int8_float32, 5s chunks)", **title_kw)
ax5.legend(fontsize=8, facecolor="#0f3460", labelcolor="white")
ax5.set_ylim(0, 5000)
ax5.grid(axis="y", color="#333", linewidth=0.5)
for i, (m, p) in enumerate(zip(stream_medians, stream_p95)):
    ax5.text(i - 0.2, m + 50, f"{m}", ha="center", color="white", fontsize=7.5)
    ax5.text(i + 0.2, p + 50, f"{p}", ha="center", color="white", fontsize=7.5)

# ── Plot 6: WER/CER accuracy (local test) ────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 0])
style_ax(ax6)

wer_labels = ["VAD on\n(file mode)", "VAD off\n(streaming)"]
wer_vals   = [0.179, 0.180]
cer_vals   = [0.084, 0.084]
x_w = np.arange(2)
ax6.bar(x_w - 0.2, wer_vals, 0.35, label="WER", color="#e74c3c", alpha=0.9)
ax6.bar(x_w + 0.2, cer_vals, 0.35, label="CER", color="#e67e22", alpha=0.9)
ax6.set_xticks(x_w)
ax6.set_xticklabels(wer_labels, color="white")
ax6.set_ylabel("Error Rate", color="white")
ax6.set_title("WER / CER — Local Test\n(221 files, CUDA int8_float32 accurate)", **title_kw)
ax6.legend(fontsize=9, facecolor="#0f3460", labelcolor="white")
ax6.set_ylim(0, 0.25)
ax6.grid(axis="y", color="#333", linewidth=0.5)
for i, (w, c) in enumerate(zip(wer_vals, cer_vals)):
    ax6.text(i - 0.2, w + 0.003, f"{w:.3f}", ha="center", color="white", fontsize=8)
    ax6.text(i + 0.2, c + 0.003, f"{c:.3f}", ha="center", color="white", fontsize=8)

# ── Plot 7: Colab sweep WER comparison ───────────────────────────────────────
ax7 = fig.add_subplot(gs[2, 1])
style_ax(ax7)

sw_labels = list(sweep_accuracy.keys())
sw_wer    = [v["wer"] for v in sweep_accuracy.values()]
sw_cer    = [v["cer"] for v in sweep_accuracy.values()]
x_sw = np.arange(len(sw_labels))
ax7.bar(x_sw - 0.2, sw_wer, 0.35, label="WER", color="#e74c3c", alpha=0.9)
ax7.bar(x_sw + 0.2, sw_cer, 0.35, label="CER", color="#e67e22", alpha=0.9)
ax7.set_xticks(x_sw)
ax7.set_xticklabels(sw_labels, color="white", fontsize=8)
ax7.set_ylabel("Error Rate", color="white")
ax7.set_title("Accuracy vs Colab Sweep\n(quantization impact)", **title_kw)
ax7.legend(fontsize=9, facecolor="#0f3460", labelcolor="white")
ax7.set_ylim(0.12, 0.22)
ax7.grid(axis="y", color="#333", linewidth=0.5)
for i, (w, c) in enumerate(zip(sw_wer, sw_cer)):
    ax7.text(i - 0.2, w + 0.001, f"{w:.3f}", ha="center", color="white", fontsize=7.5)
    ax7.text(i + 0.2, c + 0.001, f"{c:.3f}", ha="center", color="white", fontsize=7.5)

# ── Plot 8: Hallucinations comparison ─────────────────────────────────────────
ax8 = fig.add_subplot(gs[2, 2])
style_ax(ax8)

hall_labels = ["Default\nfloat16", "Default\nint8_f16", "Baseline\nint8_f16",
               "Local\nVAD on", "Local\nVAD off"]
hall_vals   = [2, 1, 0, 2, 4]
hall_colors = ["#e74c3c", "#e67e22", "#2ecc71", "#3498db", "#e74c3c"]
bars = ax8.bar(hall_labels, hall_vals, color=hall_colors, alpha=0.9)
for bar, val in zip(bars, hall_vals):
    ax8.text(bar.get_x() + bar.get_width()/2, val + 0.05,
             str(val), ha="center", color="white", fontsize=10, fontweight="bold")
ax8.set_ylabel("Hallucination count", color="white")
ax8.set_title("Hallucinations\n(>30% word inflation)", **title_kw)
ax8.set_ylim(0, 5.5)
ax8.grid(axis="y", color="#333", linewidth=0.5)
ax8.tick_params(labelsize=8)

plt.savefig(PLOT_FILE, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {PLOT_FILE}")
plt.close()
