"""Render self-contained HTML reports from sections (text, key/value grids, tables, raw HTML)."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from quant import __version__
from quant.core.time import now

TEMPLATES = Path(__file__).resolve().parent / "templates"
_env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=select_autoescape(["html"]))


@dataclass
class Section:
    title: str
    text: str | None = None
    kv: Mapping[str, object] | None = None
    table: pd.DataFrame | None = None
    html: str | None = None
    chart: Mapping[str, object] | None = (
        None  # plotly spec {data, layout}; colours assigned by role at render
    )
    chart_height: int = 320
    chart_id: str = ""
    float_format: str = "{:,.4g}"
    max_rows: int = 500
    extras: dict[str, object] = field(default_factory=dict)

    @property
    def chart_json(self) -> str | None:
        if self.chart is None:
            return None
        return json.dumps(self.chart, default=_json_default).replace("</", "<\\/")

    @property
    def table_html(self) -> str | None:
        if self.table is None:
            return None
        df = self.table.head(self.max_rows)
        fmt = self.float_format
        return df.to_html(index=False, border=0, na_rep="", float_format=lambda v: fmt.format(v), escape=True)


def plotly_js_source() -> str:
    """The plotly.js bundle shipped with the installed Python `plotly` package (for fully offline reports)."""
    from plotly.offline import get_plotlyjs

    return get_plotlyjs()


def _json_default(value: object) -> object:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, float) and value != value:
        return None
    return str(value)


def render_report(
    title: str,
    sections: Sequence[Section],
    caveats: Sequence[str],
    *,
    subtitle: str = "",
    path: Path | str | None = None,
    inline_js: str | None = None,
) -> str:
    sections = list(sections)
    for i, s in enumerate(sections):
        if s.chart is not None and not s.chart_id:
            s.chart_id = f"chart-{i}"
    html = _env.get_template("report.html").render(
        title=title,
        subtitle=subtitle,
        sections=sections,
        has_charts=any(s.chart is not None for s in sections),
        inline_js=inline_js,
        caveats=list(caveats),
        generated_at=now().strftime("%Y-%m-%d %H:%M UTC"),
        version=__version__,
    )
    if path is not None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
    return html
