"""
Ablation Study PDF Generator
==============================
Verwendung:

  Option A – Werte direkt als Dictionary übergeben:
      results = {
          "Baseline CNN": {"accuracy": 61.2, "macro_f1": 0.54, "infer_ms": 8,  "params_m": 0.4},
          "ResNet-18":    {"accuracy": 79.4, "macro_f1": 0.76, "infer_ms": 12, "params_m": 11.7},
      }
      generate_report(results, best_model="ResNet-18")

  Option B – Im Trainings-Loop befüllen:
      tracker = AblationTracker()
      # ... Training Baseline ...
      tracker.add("Baseline CNN", accuracy=61.2, macro_f1=0.54, infer_ms=8, params_m=0.4)
      # ... Training ResNet ...
      tracker.add("ResNet-18", accuracy=79.4, macro_f1=0.76, infer_ms=12, params_m=11.7)
      tracker.save_pdf("bericht.pdf")
"""

import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np

from configs.load_configs import configs
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT

# ── COLOR-CODES ─────────────────────────────────────────────────────────────────────
C_BEST    = "#1D9E75"
C_NORMAL  = "#378ADD"
C_TEXT    = "#1a1a1a"
C_MUTED   = "#6b6b6b"
C_BORDER  = "#e0e0e0"
C_STRIPE  = "#fafaf9"
C_HLBG    = "#f0faf6"
C_HEADBG  = "#f4f4f2"


class AblationTracker:
    """Collects data and creates a report with collected data"""

    def __init__(self):
        self.results = {}
        self._order  = []

    def add(self, model_name: str, accuracy: float, f1_score: float, loss: float, params_m: float):
        """
        Adds data of a trained model

        Parameters
        ----------
        model_name: str
            name of the model

        accuracy: float
            test accuracy 

        f1_score: float
            the calculated F1 Score 
        
        loss : float
            test loss 

        params_m: float
            the number of parameters in millions
        """
        self.results[model_name] = {
            "accuracy": accuracy,
            "f1_score": f1_score,
            "loss"    : loss,
            "params_m": params_m
        }
        if model_name not in self._order:
            self._order.append(model_name)

    def save_pdf(self):

        ordered = {k: self.results[k] for k in self._order}
        generate_report(ordered, configs["ablation_study_path"], configs["ablation_study_title"])

def generate_report(results: dict, output_path: str = "ablation_report.pdf", title: str = "Modellvergleich — Ablation Study"):
    """
    Creates the results pdf based of the results dictionary

    Parameters
    ----------
        results: dict
            {model_name : str : {accuracy: float, f1_score: float, params_m: float}}

        output_path: str
            filepath for the .pdf file that should be cretaed

        title: str
            title of the .pdf file that should be created
    """
    model_names     = list(results.keys())
    
    accuracies = []
    f1_scores  = []
    losses     = []
    params_m   = []

    for model_name in model_names:

        accuracies.append(results[model_name]["accuracy"])
        f1_scores.append(results[model_name]["f1_score"])
        losses.append(results[model_name]["loss"])
        params_m.append(results[model_name]["params_m"])

    best_model = model_names[f1_scores.index(max(f1_scores))]

    bar_colors = []

    for current_model_name in model_names:

        if current_model_name in best_model:
            bar_colors.append(C_BEST)
        else:
            bar_colors.append(C_NORMAL)

    chart_buf = _build_charts(model_names, accuracies, f1_scores, losses, bar_colors, title)
    _build_pdf(output_path, title, chart_buf, model_names, accuracies, losses, f1_scores, params_m, best_model, bar_colors)
    print(f"[+] Report saved at: {output_path}")



def _build_charts(model_names : list, accuracies : list, losses : list, f1_scores : list , bar_colors : list , title : str) -> ptr:
    short = [model_name.replace(" + ", "\n+ ") for model_name in model_names]

    fig = plt.figure(figsize=(11, 5.5), facecolor="white")
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38, left=0.06, right=0.97, top=0.85, bottom=0.22)

    def style_ax(ax, ylabel):
        ax.set_facecolor("white")
        ax.spines[["top","right","left"]].set_visible(False)
        ax.spines["bottom"].set_color(C_BORDER)
        ax.tick_params(axis="y", length=0, labelsize=8.5, labelcolor=C_MUTED)
        ax.tick_params(axis="x", length=0, labelsize=8, labelcolor=C_TEXT)
        ax.set_ylabel(ylabel, fontsize=9, color=C_MUTED, labelpad=6)
        ax.yaxis.grid(True, color="#f0f0f0", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)

    x = range(len(model_names))

    # f1-score
    ax1 = fig.add_subplot(gs[0])
    bars1 = ax1.bar(x, f1_scores, color = bar_colors, width = 0.55, zorder = 3, linewidth = 0)
    for bar, val, color in zip(bars1, f1_scores, bar_colors):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.012, f"{val:.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=C_BEST if color == C_BEST else C_MUTED)

    ax1.set_xticks(x)
    ax1.set_xticklabels(short, fontsize=8)
    ax1.set_ylim(0, 1.12)
    ax1.set_title("F1-Scores", fontsize=10, fontweight="bold", color=C_TEXT, pad=8, loc="left")
    style_ax(ax1, "F1-Score")


    # loss
    ax2 = fig.add_subplot(gs[1])
    bars2 = ax2.bar(x, losses, color = bar_colors, width = 0.55, zorder = 3, linewidth = 0)
    for bar, val, color in zip(bars2, losses, bar_colors):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.012,f"{val:.2f}", ha = "center", va="bottom", fontsize=8.5, fontweight="bold", color=C_BEST if color == C_BEST else C_MUTED)

    ax2.set_xticks(x)
    ax2.set_xticklabels(short, fontsize=8)
    ax2.set_ylim(0, 1.12)
    ax2.set_title("Losses", fontsize=10, fontweight="bold", color=C_TEXT, pad=8, loc="left")
    style_ax(ax2, "Loss")
    
    # accuracy
    ax3 = fig.add_subplot(gs[2])
    bars3 = ax3.bar(x, accuracies, color=bar_colors, width=0.55, zorder=3, linewidth=0)
    for bar, val, color in zip(bars3, accuracies, bar_colors):
        ax3.text(bar.get_x() + bar.get_width()/2, val + 0.012, f"{val * 100:.2f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=C_BEST if color == C_BEST else C_MUTED)
    
    ax3.set_xticks(x)
    ax3.set_xticklabels(short, fontsize=8)
    ax3.set_ylim(0, 115)
    ax3.set_title("Accuracies", fontsize=10, fontweight="bold", color=C_TEXT, pad=8, loc="left")
    style_ax(ax3, "Accuracy")
    
    patch = mpatches.Patch(color=C_BEST, label="Best Model")
    fig.legend(handles=[patch], loc="upper right", fontsize=8.5, frameon=False, labelcolor=C_TEXT)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    plt.close(fig)
    return buf


def _build_pdf(output_path, title, chart_buf, model_names, accuracies, losses, f1_scores,  params_m, best_model, bar_colors):

    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm,  bottomMargin=2*cm,
    )

    title_style = ParagraphStyle("T", fontSize=17, fontName="Helvetica-Bold", textColor=colors.HexColor(C_TEXT), spaceAfter=4)
    sub_style   = ParagraphStyle("S", fontSize=9,  fontName="Helvetica", textColor=colors.HexColor(C_MUTED), spaceAfter=14)
    sec_style   = ParagraphStyle("H", fontSize=11, fontName="Helvetica-Bold", textColor=colors.HexColor(C_TEXT), spaceBefore=12, spaceAfter=8)
    note_style  = ParagraphStyle("N", fontSize=8,  fontName="Helvetica", textColor=colors.HexColor(C_MUTED))

    story = []
    story.append(Paragraph(title, title_style))
    story.append(Paragraph( f"All models have been trained on the same dataset with the same seed, the datasplit was train: {configs["train_ratio"]} , val: {configs["val_ratio"]} , test: {configs["test_ratio"]}" , sub_style))

    # Diagramme
    story.append(Image(chart_buf, width=16.5*cm, height=9*cm))
    story.append(Spacer(1, 0.3*cm))

    # Tabelle
    story.append(Paragraph("Detaillierte Ergebnisse", sec_style))

    header = ["Modell", "Top-1 Acc.", "Loss", "F1-Score", "Parameter"]
    rows   = [header]
    best_row_idx = None

    for i, model_name in enumerate(model_names):
        if(model_name == best_model):
            marker = " ✓"
        
        else:
            marker = ""
        rows.append([
            model_name + marker,
            f"{accuracies[i]:.2f}",
            f"{losses[i]:.2f}",
            f"{f1_scores[i]:.2f}",
            f"{params_m[i]:.1f} M"
        ])
        if model_name == best_model:
            best_row_idx = i + 1  # +1 wegen Header

    col_w = [7.5*cm, 2.3*cm, 2.3*cm, 2.2*cm, 2.4*cm]
    tbl   = Table(rows, colWidths=col_w, repeatRows=1)

    ts = [
        ("BACKGROUND",    (0,0), (-1,0),  colors.HexColor(C_HEADBG)),
        ("TEXTCOLOR",     (0,0), (-1,0),  colors.HexColor(C_MUTED)),
        ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 9),
        ("TOPPADDING",    (0,0), (-1,-1), 7),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7),
        ("ALIGN",         (1,0), (-1,-1), "CENTER"),
        ("ALIGN",         (0,0), (0,-1),  "LEFT"),
        ("FONTNAME",      (0,1), (-1,-1), "Helvetica"),
        ("TEXTCOLOR",     (0,1), (-1,-1), colors.HexColor(C_TEXT)),
        ("LINEBELOW",     (0,0), (-1,0),  0.8, colors.HexColor(C_BORDER)),
        ("LINEBELOW",     (0,1), (-1,-2), 0.4, colors.HexColor("#ebebeb")),
        ("BOX",           (0,0), (-1,-1), 0.5, colors.HexColor(C_BORDER)),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.white, colors.HexColor(C_STRIPE)]),
    ]

    # Hughlight best column
    if best_row_idx is not None:
        ts += [
            ("BACKGROUND", (0, best_row_idx), (-1, best_row_idx), 
            colors.HexColor(C_HLBG)),
            ("TEXTCOLOR",  (1, best_row_idx), (-1, best_row_idx),
            colors.HexColor(C_BEST)),
            ("FONTNAME",   (0, best_row_idx), (-1, best_row_idx),
            "Helvetica-Bold"),
        ]

    tbl.setStyle(TableStyle(ts))
    story.append(tbl)
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        "Due to inequalities between the number of samples in every class the F1-Score is being used to determine the performance of a model"
        "F1-Score weighs every class equally and therefore prevents class distortion from inequality between classes",
        note_style))

    doc.build(story)


# ══════════════════════════════════════════════════════════════════════════════
# Example Usage
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    results = {
        "Simples CNN (Baseline)":            {"accuracy": 61.2, "macro_f1": 0.54, "params_m": 0.4},
        "ResNet-18":                         {"accuracy": 79.4, "macro_f1": 0.76, "params_m": 11.7},
        "ResNet-18 + Augmentation":          {"accuracy": 84.1, "macro_f1": 0.82, "params_m": 11.7},
        "ResNet-18 + Aug. + Weighted Loss":  {"accuracy": 87.3, "macro_f1": 0.86, "params_m": 11.7},
    }
    generate_report(results, output_path="/mnt/user-data/outputs/ablation_report.pdf")

