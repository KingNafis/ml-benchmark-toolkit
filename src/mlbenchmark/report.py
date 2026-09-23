"""Standalone HTML/Markdown report generation for model comparisons."""

from __future__ import annotations

import base64
import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from matplotlib.figure import Figure

from .comparator import ModelComparator


def _figure_to_base64(fig: Figure) -> str:
    """Encode a matplotlib Figure as a base64 PNG data URI for HTML embedding."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=130, bbox_inches="tight")
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    buffer.close()
    return f"data:image/png;base64,{encoded}"


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    margin: 0; padding: 2rem; background: #f6f7f9; color: #1a1a1a;
  }}
  .container {{ max-width: 1100px; margin: 0 auto; }}
  h1 {{ font-size: 1.8rem; margin-bottom: 0.2rem; }}
  .subtitle {{ color: #666; margin-bottom: 2rem; font-size: 0.9rem; }}
  h2 {{ border-bottom: 2px solid #e0e0e0; padding-bottom: 0.4rem; margin-top: 2.5rem; }}
  table {{ border-collapse: collapse; width: 100%; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  th, td {{ border: 1px solid #e0e0e0; padding: 8px 12px; text-align: right; font-size: 0.9rem; }}
  th {{ background: #2b3a55; color: white; text-align: center; }}
  td:first-child, th:first-child {{ text-align: left; }}
  tr:nth-child(even) {{ background: #fafbfc; }}
  tr:first-child td {{ font-weight: 600; background: #eef4ff; }}
  .fig-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 1.5rem; margin-top: 1rem; }}
  .fig-card {{ background: white; border-radius: 8px; padding: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  .fig-card img {{ width: 100%; height: auto; border-radius: 4px; }}
  .fig-card h3 {{ margin: 0 0 0.5rem 0; font-size: 1rem; color: #2b3a55; }}
  footer {{ margin-top: 3rem; color: #999; font-size: 0.8rem; text-align: center; }}
</style>
</head>
<body>
<div class="container">
  <h1>{title}</h1>
  <p class="subtitle">Generated {timestamp} &middot; ml-benchmark-toolkit</p>

  <h2>Model Comparison</h2>
  {comparison_table}

  <h2>Confusion Matrices</h2>
  <div class="fig-grid">{confusion_figs}</div>

  {roc_section}

  {training_section}

  <footer>Report generated with mlbenchmark.Report</footer>
</div>
</body>
</html>
"""


class Report:
    """Compiles a :class:`ModelComparator`'s tables and figures into a
    standalone HTML or Markdown summary dashboard.

    Examples
    --------
    >>> report = Report(comparator, title="Model Benchmark")
    >>> report.to_html("report.html")
    >>> report.to_markdown("report.md")
    """

    def __init__(
        self,
        comparator: ModelComparator,
        title: str = "Model Benchmark Report",
        rank_by: str = "f1_macro",
        normalize_confusion: bool = False,
    ) -> None:
        self.comparator = comparator
        self.title = title
        self.rank_by = rank_by
        self.normalize_confusion = normalize_confusion

    def _confusion_figs(self) -> Dict[str, Figure]:
        return self.comparator.confusion_matrices(normalize=self.normalize_confusion)

    def to_html(self, path: str) -> Path:
        """Render the full dashboard (tables + embedded figures) to a standalone HTML file."""
        table_df = self.comparator.comparison_table(rank_by=self.rank_by).round(4)
        table_html = table_df.to_html(classes="comparison-table", border=0)

        confusion_figs = self._confusion_figs()
        confusion_html = "".join(
            f'<div class="fig-card"><h3>{name}</h3>'
            f'<img src="{_figure_to_base64(fig)}" alt="Confusion matrix for {name}"></div>'
            for name, fig in confusion_figs.items()
        )

        roc_figs = self.comparator.roc_curves()
        roc_section = ""
        if roc_figs:
            roc_html = "".join(
                f'<div class="fig-card"><h3>{name}</h3>'
                f'<img src="{_figure_to_base64(fig)}" alt="ROC curve for {name}"></div>'
                for name, fig in roc_figs.items()
            )
            roc_section = f'<h2>ROC Curves</h2><div class="fig-grid">{roc_html}</div>'

        training_figs = self.comparator.training_curves()
        training_section = ""
        if training_figs:
            training_html = "".join(
                f'<div class="fig-card"><h3>{name}</h3>'
                f'<img src="{_figure_to_base64(fig)}" alt="Training curves for {name}"></div>'
                for name, fig in training_figs.items()
            )
            training_section = (
                f'<h2>Training Curves</h2><div class="fig-grid">{training_html}</div>'
            )

        html = _HTML_TEMPLATE.format(
            title=self.title,
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            comparison_table=table_html,
            confusion_figs=confusion_html,
            roc_section=roc_section,
            training_section=training_section,
        )

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html, encoding="utf-8")
        return out_path

    def to_markdown(self, path: Optional[str] = None) -> str:
        """Render the comparison table (and notes on available figures) as Markdown.

        Parameters
        ----------
        path : str, optional
            If provided, the Markdown is also written to this file path.
        """
        table_df = self.comparator.comparison_table(rank_by=self.rank_by).round(4)
        lines = [
            f"# {self.title}",
            "",
            f"_Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_",
            "",
            "## Model Comparison",
            "",
            table_df.to_markdown(),
            "",
            "## Figures",
            "",
            (
                "Confusion matrices, ROC curves, and training curves are available "
                "via `Report.to_html()` or by calling the plotting functions in "
                "`mlbenchmark.plots` / `ModelComparator` directly (Markdown does not "
                "support embedded raster images inline in this export)."
            ),
        ]
        markdown = "\n".join(lines)

        if path is not None:
            out_path = Path(path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(markdown, encoding="utf-8")

        return markdown
