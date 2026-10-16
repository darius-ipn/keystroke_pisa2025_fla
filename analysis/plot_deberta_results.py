import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# CONFIG — change this to switch between conditions
# ============================================================
CONDITION = "withchat"   # options: "combined", "wo_chat", "withchat"

XLSX_PATH = Path("results_deberta.xlsx")
OUTPUT_DIR = Path("plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================
def load_condition(xlsx_path: Path, condition: str):
    """Read the comparison sheet and extract data for the given condition."""
    df = pd.read_excel(xlsx_path, sheet_name="comparison", header=None)

    # Column layout:
    # 0=condition, 1=source_timepoint, 2=n, 3=FULL QWK, 4=HYBRID QWK,
    # 5=hybrid change, 6=FUSION QWK, 7=fusion change to full

    # Forward-fill the condition column (merged cells come in as NaN)
    df[0] = df[0].ffill()

    # Drop header rows (rows where col 1 is not a timepoint)
    timepoint_rows = df[df[1].isin(["05min","10min","15min","20min","25min","full"])]

    # Filter by condition
    cond_df = timepoint_rows[timepoint_rows[0] == condition].copy()

    if cond_df.empty:
        raise ValueError(
            f"Condition '{condition}' not found. "
            f"Available: {timepoint_rows[0].unique().tolist()}"
        )

    full_qwk   = cond_df[3].astype(float).tolist()
    hybrid_qwk = cond_df[4].astype(float).tolist()
    fusion_qwk = cond_df[6].astype(float).tolist()

    return full_qwk, hybrid_qwk, fusion_qwk


full_qwk, hybrid_qwk, fusion_qwk = load_condition(XLSX_PATH, CONDITION)

x_labels = ["5 min", "10 min", "15 min", "20 min", "25 min", "Full"]
x = np.arange(len(x_labels))

condition_label = {
    "combined": "Combined",
    "wo_chat":  "Without Chat",
    "withchat": "With Chat",
}.get(CONDITION, CONDITION)

print(f"Condition : {CONDITION}  ({condition_label})")
print(f"FULL      : {full_qwk}")
print(f"HYBRID    : {hybrid_qwk}")
print(f"FUSION    : {fusion_qwk}")


# ============================================================
# VERSION 1: SIMPLE — clean, minimal, publication-ready
# ============================================================
def plot_simple():
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(x, full_qwk,   'o-', color='#4477AA', linewidth=2, markersize=7, label='FULL (text-only)')
    ax.plot(x, hybrid_qwk, 's-', color='#EE6677', linewidth=2, markersize=7, label='HYBRID')
    ax.plot(x, fusion_qwk, 'D-', color='#228833', linewidth=2, markersize=7, label='COMBI')

    ax.set_xlabel('Writing Time Threshold', fontsize=12)
    ax.set_ylabel('QWK', fontsize=12)
    ax.set_title(f'Model Performance Over Writing Time ({condition_label})', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"plot_simple_{CONDITION}.png", dpi=200, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / f"plot_simple_{CONDITION}.pdf", bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: plot_simple_{CONDITION}.png / .pdf")


# ============================================================
# VERSION 2: COOL — dark theme with glow effect
# ============================================================
def plot_cool():
    plt.rcParams.update({'text.color': 'white', 'axes.labelcolor': 'white',
                         'xtick.color': 'white', 'ytick.color': 'white'})

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    colors  = ['#00d2ff', '#ff6b6b', '#51cf66']
    labels  = ['FULL (text-only)', 'HYBRID', 'COMBI']
    markers = ['o', 's', 'D']
    data    = [full_qwk, hybrid_qwk, fusion_qwk]

    for vals, col, lab, mk in zip(data, colors, labels, markers):
        ax.plot(x, vals, color=col, linewidth=6, alpha=0.15)
        ax.plot(x, vals, color=col, linewidth=3, alpha=0.4)
        ax.plot(x, vals, marker=mk, color=col, linewidth=2, markersize=9,
                label=lab, markeredgecolor='white', markeredgewidth=0.8, zorder=5)
        for xi, yi in zip(x, vals):
            ax.annotate(f'{yi:.2f}', (xi, yi), textcoords="offset points",
                        xytext=(0, 12), ha='center', fontsize=8,
                        color=col, fontweight='bold')

    ax.fill_between(x, full_qwk, fusion_qwk, alpha=0.08, color='#51cf66')

    ax.set_xlabel('Writing Time Threshold', fontsize=13, fontweight='bold')
    ax.set_ylabel('QWK', fontsize=13, fontweight='bold')
    ax.set_title(f'Model Performance Over Writing Time ({condition_label})',
                 fontsize=16, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=11)
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    ax.grid(True, alpha=0.15, color='white', linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#444')
    ax.spines['bottom'].set_color('#444')

    legend = ax.legend(frameon=True, fontsize=11, loc='lower right',
                       facecolor='#16213e', edgecolor='#444')
    for text in legend.get_texts():
        text.set_color('white')

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"plot_cool_{CONDITION}.png", dpi=200,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    fig.savefig(OUTPUT_DIR / f"plot_cool_{CONDITION}.pdf",
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    plt.rcParams.update(plt.rcParamsDefault)
    print(f"Saved: plot_cool_{CONDITION}.png / .pdf")


# ============================================================
# VERSION 3: SMART — main panel + ΔQWK bar panel
# ============================================================
def plot_smart():
    fig, (ax_main, ax_delta) = plt.subplots(
        2, 1, figsize=(10, 7.5),
        gridspec_kw={'height_ratios': [3, 1.2]},
        sharex=True
    )

    colors = {'full': '#3366CC', 'hybrid': '#DC3912', 'fusion': '#109618'}

    # --- Main panel ---
    ax_main.plot(x, full_qwk,   'o-', color=colors['full'],   linewidth=2.2, markersize=8,
                 label='FULL (text-only)', zorder=5)
    ax_main.plot(x, hybrid_qwk, 's-', color=colors['hybrid'], linewidth=2.2, markersize=8,
                 label='HYBRID', zorder=5)
    ax_main.plot(x, fusion_qwk, 'D-', color=colors['fusion'], linewidth=2.2, markersize=8,
                 label='COMBI', zorder=5)

    # Find the timepoint where FUSION advantage is largest
    fusion_advantage = [fu - f for fu, f in zip(fusion_qwk, full_qwk)]
    peak_idx = int(np.argmax(fusion_advantage))

    ax_main.axvspan(-0.3, peak_idx + 0.5, alpha=0.06, color=colors['fusion'], zorder=0)
    ax_main.annotate('Keystroke features\nmost impactful here',
                     xy=(peak_idx / 2, min(full_qwk) + 0.05),
                     fontsize=9, fontstyle='italic', color='#555', ha='center')

    # Convergence annotation at "full"
    ax_main.annotate('Models converge\nat full text',
                     xy=(5, max(full_qwk)), xytext=(3.8, max(full_qwk) - 0.2),
                     fontsize=9, fontstyle='italic', color='#555',
                     arrowprops=dict(arrowstyle='->', color='#999', lw=1.2),
                     ha='center')

    ax_main.set_ylabel('QWK', fontsize=12, fontweight='bold')
    ax_main.set_title(f'Model Performance Over Writing Time ({condition_label})',
                      fontsize=14, fontweight='bold', pad=12)
    ax_main.set_ylim(0.0, 1.05)
    ax_main.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    ax_main.legend(frameon=True, fontsize=10, loc='lower right')
    ax_main.grid(True, alpha=0.2)
    ax_main.spines['top'].set_visible(False)
    ax_main.spines['right'].set_visible(False)

    # --- Delta panel ---
    delta_hybrid = [h - f for h, f in zip(hybrid_qwk, full_qwk)]
    delta_fusion = [fu - f for fu, f in zip(fusion_qwk, full_qwk)]

    ax_delta.bar(x - 0.15, delta_hybrid, width=0.3, color=colors['hybrid'],
                 alpha=0.8, label='HYBRID − FULL')
    ax_delta.bar(x + 0.15, delta_fusion, width=0.3, color=colors['fusion'],
                 alpha=0.8, label='COMBI − FULL')

    ax_delta.axhline(0, color='black', linewidth=0.8)
    ax_delta.set_ylabel('ΔQWK vs FULL', fontsize=11, fontweight='bold')
    ax_delta.set_xlabel('Writing Time Threshold', fontsize=12, fontweight='bold')
    ax_delta.set_xticks(x)
    ax_delta.set_xticklabels(x_labels, fontsize=11)
    y_abs = max(abs(v) for v in delta_hybrid + delta_fusion) + 0.05
    ax_delta.set_ylim(-y_abs, y_abs * 1.5)
    ax_delta.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    ax_delta.legend(frameon=True, fontsize=9, loc='upper right')
    ax_delta.grid(True, alpha=0.2, axis='y')
    ax_delta.spines['top'].set_visible(False)
    ax_delta.spines['right'].set_visible(False)

    fig.tight_layout(h_pad=1.5)
    fig.savefig(OUTPUT_DIR / f"plot_smart_{CONDITION}.png", dpi=200, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / f"plot_smart_{CONDITION}.pdf", bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: plot_smart_{CONDITION}.png / .pdf")


# --- Run all ---
plot_simple()
plot_cool()
plot_smart()
print(f"\nAll plots saved to {OUTPUT_DIR}")