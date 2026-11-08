"""Render self-contained HTML reports from sections (text, key/value grids, tables, raw HTML)."""

from __future__ import annotations

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
    float_format: str = "{:,.4g}"
    max_rows: int = 500
    extras: dict[str, object] = field(default_factory=dict)

    @property
    def table_html(self) -> str | None:
        if self.table is None:
            return None
        df = self.table.head(self.max_rows)
        fmt = self.float_format
        return df.to_html(index=False, border=0, na_rep="", float_format=lambda v: fmt.format(v), escape=True)


def render_report(
    title: str,
    sections: Sequence[Section],
    caveats: Sequence[str],
    *,
    subtitle: str = "",
    path: Path | str | None = None,
) -> str:
    html = _env.get_template("report.html").render(
        title=title,
        subtitle=subtitle,
        sections=list(sections),
        caveats=list(caveats),
        generated_at=now().strftime("%Y-%m-%d %H:%M UTC"),
        version=__version__,
    )
    if path is not None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
    return html
