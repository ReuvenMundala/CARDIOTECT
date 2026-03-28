import json
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import os
from reportlab.lib import colors # type: ignore
from reportlab.lib.pagesizes import letter # type: ignore
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image # type: ignore
from reportlab.lib.styles import getSampleStyleSheet # type: ignore, ParagraphStyle

def save_json_report(metrics, path):
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=4)

def save_csv_summary(metrics_list, path):
    df = pd.DataFrame(metrics_list)
    df.to_csv(path, index=False)

def generate_pdf_report(scores_pred, scores_gt=None, plots_paths=None, output_path="report.pdf"):
    """
    Generates a PDF report using ReportLab.
    scores_pred: dict of AI scores
    scores_gt: dict of Expert scores (optional)
    plots_paths: list of paths to image files (plots)
    """
    if plots_paths is None:
        plots_paths = []
        
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    title_style = styles['Title']
    story.append(Paragraph("Cardiotect Calcium Scoring Report", title_style))
    story.append(Spacer(1, 12))

    # Metrics Table
    if scores_gt:
        headers = ["Metric", "Expert (GT)", "AI (Pred)"]
        data = [headers]
        keys = list(scores_pred.keys())
        for k in keys:
            v_ai = scores_pred.get(k, "N/A")
            v_gt = scores_gt.get(k, "N/A")
            
            if isinstance(v_ai, (int, float)): v_ai = f"{v_ai:.1f}"
            if isinstance(v_gt, (int, float)): v_gt = f"{v_gt:.1f}"
            
            data.append([str(k), str(v_gt), str(v_ai)])
    else:
        headers = ["Metric", "Value"]
        data = [headers]
        for k, v in scores_pred.items():
            if isinstance(v, (int, float)):
                v = f"{v:.1f}"
            data.append([str(k), str(v)])

    t = Table(data)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(t)
    story.append(Spacer(1, 12))

    # Plots
    for p_path in plots_paths:
        if os.path.exists(p_path):
            img = Image(p_path, width=400, height=300)
            story.append(img)
            story.append(Spacer(1, 12))

    doc.build(story)
