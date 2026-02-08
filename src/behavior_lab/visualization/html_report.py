"""Self-contained HTML report generator for behavior analysis pipelines.

Produces a single HTML file with all images base64-embedded,
tab navigation per dataset, and responsive CSS.
"""
from __future__ import annotations

import base64
import time
from io import BytesIO
from pathlib import Path
from typing import Any


def fig_to_base64(fig, dpi: int = 150) -> str:
    """Convert a matplotlib Figure to a base64 data URI (PNG).

    Args:
        fig: matplotlib Figure object
        dpi: Resolution

    Returns:
        Data URI string: "data:image/png;base64,..."
    """
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    buf.close()
    return f"data:image/png;base64,{b64}"


def image_to_base64(path: str | Path) -> str:
    """Read an image file and return a base64 data URI.

    Supports PNG, JPG, GIF.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
    }.get(suffix, "image/png")

    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _escape(text: str) -> str:
    """Basic HTML escape."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _render_metric_card(label: str, value: Any) -> str:
    if isinstance(value, float):
        display = f"{value:.4f}"
    else:
        display = str(value)
    return f"""<div class="metric-card">
  <div class="metric-value">{_escape(display)}</div>
  <div class="metric-label">{_escape(label)}</div>
</div>"""


def _render_table(headers: list[str], rows: list[list[Any]]) -> str:
    ths = "".join(f"<th>{_escape(h)}</th>" for h in headers)
    trs = []
    for row in rows:
        tds = "".join(f"<td>{_escape(str(v))}</td>" for v in row)
        trs.append(f"<tr>{tds}</tr>")
    return f"""<table>
<thead><tr>{ths}</tr></thead>
<tbody>{"".join(trs)}</tbody>
</table>"""


def _render_image(src: str, alt: str = "", max_width: int = 600) -> str:
    return f'<img src="{src}" alt="{_escape(alt)}" style="max-width:{max_width}px;width:100%;border-radius:8px;margin:8px 0;">'


def _render_dataset_tab(name: str, ds: dict) -> str:
    """Render the content for one dataset tab."""
    sections: list[str] = []

    # Data summary
    data = ds.get("data", {})
    if data:
        rows = [[k, str(v)] for k, v in data.items()]
        sections.append(f"<h3>Data Summary</h3>{_render_table(['Field', 'Value'], rows)}")

    # Figures
    figures = ds.get("figures", {})
    for fig_name, fig_src in figures.items():
        label = fig_name.replace("_", " ").title()
        if fig_src:
            sections.append(f"<h3>{_escape(label)}</h3>{_render_image(fig_src, alt=label)}")

    # Metrics
    metrics = ds.get("metrics", {})
    if metrics:
        cards = "".join(_render_metric_card(k, v) for k, v in metrics.items())
        sections.append(f"<h3>Metrics</h3><div class='metrics-grid'>{cards}</div>")

    # Cluster metrics
    cluster_metrics = ds.get("cluster_metrics", {})
    if cluster_metrics:
        cards = "".join(_render_metric_card(k, v) for k, v in cluster_metrics.items())
        sections.append(f"<h3>Cluster Metrics</h3><div class='metrics-grid'>{cards}</div>")

    # Behavior metrics
    behavior_metrics = ds.get("behavior_metrics", {})
    if behavior_metrics:
        display_items = {
            k: v for k, v in behavior_metrics.items()
            if k != "bout_durations"
        }
        cards = "".join(_render_metric_card(k, v) for k, v in display_items.items())
        sections.append(f"<h3>Behavior Metrics</h3><div class='metrics-grid'>{cards}</div>")

    # Linear probe
    probe = ds.get("linear_probe", {})
    if probe:
        cards = "".join(_render_metric_card(k, v) for k, v in probe.items())
        sections.append(f"<h3>Linear Probe</h3><div class='metrics-grid'>{cards}</div>")

    return "\n".join(sections)


_CSS = """
:root {
  --bg: #f7f8fa;
  --card-bg: #ffffff;
  --primary: #2c3e50;
  --accent: #3498db;
  --accent2: #1abc9c;
  --text: #2c3e50;
  --text-light: #7f8c8d;
  --border: #e1e8ed;
  --radius: 10px;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: var(--bg); color: var(--text);
  line-height: 1.6; padding: 0;
}
.header {
  background: linear-gradient(135deg, var(--primary), var(--accent));
  color: white; padding: 32px 40px; margin-bottom: 24px;
}
.header h1 { font-size: 1.8em; font-weight: 600; }
.header .subtitle { opacity: 0.85; margin-top: 4px; }
.container { max-width: 1200px; margin: 0 auto; padding: 0 24px 40px; }
.tabs {
  display: flex; gap: 4px; border-bottom: 2px solid var(--border);
  margin-bottom: 24px; flex-wrap: wrap;
}
.tab {
  padding: 10px 20px; cursor: pointer; border: none; background: none;
  font-size: 0.95em; color: var(--text-light); border-bottom: 3px solid transparent;
  transition: all 0.2s;
}
.tab:hover { color: var(--accent); }
.tab.active { color: var(--accent); border-bottom-color: var(--accent); font-weight: 600; }
.tab-content { display: none; animation: fadeIn 0.3s ease; }
.tab-content.active { display: block; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
h3 { margin: 24px 0 12px; color: var(--primary); font-size: 1.15em; }
table {
  width: 100%; border-collapse: collapse; margin: 12px 0;
  background: var(--card-bg); border-radius: var(--radius);
  overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}
th { background: var(--primary); color: white; padding: 10px 14px; text-align: left; font-weight: 500; }
td { padding: 8px 14px; border-bottom: 1px solid var(--border); }
tr:hover td { background: #f0f4f8; }
.metrics-grid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 12px; margin: 12px 0;
}
.metric-card {
  background: var(--card-bg); border-radius: var(--radius);
  padding: 16px; text-align: center;
  box-shadow: 0 1px 3px rgba(0,0,0,0.08);
  transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.12); }
.metric-value { font-size: 1.4em; font-weight: 700; color: var(--accent); }
.metric-label { font-size: 0.82em; color: var(--text-light); margin-top: 4px; }
.overview-stats {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
  gap: 16px; margin: 16px 0;
}
.stat-card {
  background: var(--card-bg); border-radius: var(--radius); padding: 20px;
  border-left: 4px solid var(--accent2);
  box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}
.stat-card h4 { color: var(--accent2); margin-bottom: 8px; }
img { border: 1px solid var(--border); }
footer {
  text-align: center; padding: 24px; color: var(--text-light);
  font-size: 0.85em; border-top: 1px solid var(--border); margin-top: 40px;
}
"""

_JS = """
function switchTab(evt, tabId) {
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById(tabId).classList.add('active');
  evt.currentTarget.classList.add('active');
}
"""


def generate_pipeline_report(
    report_data: dict,
    output_path: str | Path,
    title: str = "Behavior Analysis Report",
) -> Path:
    """Generate a self-contained HTML report.

    Args:
        report_data: Dictionary with structure:
            {
                "title": str (optional),
                "timestamp": str (optional),
                "datasets": {
                    "calms21": {
                        "data": {...},
                        "figures": {"name": "data:image/..."},  # base64 URIs
                        "metrics": {...},
                        "cluster_metrics": {...},
                        "behavior_metrics": {...},
                        "linear_probe": {...},
                    },
                    ...
                }
            }
        output_path: Where to write the HTML file
        title: Report title

    Returns:
        Path to the generated HTML file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rpt_title = report_data.get("title", title)
    timestamp = report_data.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
    datasets = report_data.get("datasets", {})

    # Build tab buttons and content
    tab_buttons: list[str] = []
    tab_contents: list[str] = []

    # Overview tab
    tab_buttons.append(
        '<button class="tab active" onclick="switchTab(event, \'tab-overview\')">Overview</button>'
    )
    overview_cards = []
    for ds_name, ds in datasets.items():
        data = ds.get("data", {})
        shape = data.get("shape", "N/A")
        n_train = data.get("n_train", "?")
        n_test = data.get("n_test", "?")
        overview_cards.append(
            f"""<div class="stat-card">
  <h4>{_escape(ds_name.upper())}</h4>
  <p>Train: <strong>{n_train}</strong> | Test: <strong>{n_test}</strong></p>
  <p>Shape: <code>{_escape(str(shape))}</code></p>
</div>"""
        )

    overview_html = f"""<div id="tab-overview" class="tab-content active">
<h3>Dataset Overview</h3>
<div class="overview-stats">{"".join(overview_cards)}</div>
</div>"""
    tab_contents.append(overview_html)

    # Per-dataset tabs
    for ds_name, ds in datasets.items():
        tab_id = f"tab-{ds_name}"
        display_name = ds_name.upper().replace("_", " ")
        tab_buttons.append(
            f'<button class="tab" onclick="switchTab(event, \'{tab_id}\')">{_escape(display_name)}</button>'
        )
        content = _render_dataset_tab(ds_name, ds)
        tab_contents.append(
            f'<div id="{tab_id}" class="tab-content">{content}</div>'
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{_escape(rpt_title)}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="header">
  <h1>{_escape(rpt_title)}</h1>
  <div class="subtitle">Generated: {_escape(timestamp)}</div>
</div>
<div class="container">
  <div class="tabs">{"".join(tab_buttons)}</div>
  {"".join(tab_contents)}
</div>
<footer>behavior-lab &middot; Generated with Python</footer>
<script>{_JS}</script>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    return output_path
