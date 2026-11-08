"""Index constituents from free sources (Wikipedia), with shape checks and provenance.

Free data gives *current* membership and, for the S&P 500, a "selected changes" table from which history
can be reconstructed approximately (incomplete before ~2008, keyed on tickers that get reused). The FTSE
pages carry no usable changes table, so FTSE universes stay `current_only` until snapshots accumulate.
Every parse records the page's SHA-256 and fails loudly when the table shape drifts.

US equities take the Nasdaq calendar family (XNAS) when they are in the Nasdaq-100 snapshot and XNYS
otherwise. Both calendars are identical; the market here is a calendar family and an id namespace, not a
verified listing venue. Existing catalogue instruments are reused by Yahoo symbol so ids stay stable.
"""

from __future__ import annotations

import hashlib
import io
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from quant.core.catalog import InstrumentCatalog
from quant.core.errors import UniverseError
from quant.core.time import today
from quant.core.types import AssetClass, Currency, Instrument, Market, StampDutyClass
from quant.data.adapters.base import BaseHTTPAdapter, Capability, RateLimit


@dataclass(frozen=True)
class UniverseSpec:
    name: str
    url: str
    expected_rows: tuple[int, int]
    symbol_cols: tuple[str, ...]
    name_cols: tuple[str, ...]
    sector_cols: tuple[str, ...]
    market: Market
    has_changes: bool = False


UNIVERSES: dict[str, UniverseSpec] = {
    "sp500": UniverseSpec(
        "sp500",
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        (480, 520),
        ("Symbol", "Ticker"),
        ("Security", "Company"),
        ("GICS Sector", "Sector"),
        Market.XNYS,
        has_changes=True,
    ),
    "ndx100": UniverseSpec(
        "ndx100",
        "https://en.wikipedia.org/wiki/Nasdaq-100",
        (95, 110),
        ("Ticker", "Symbol"),
        ("Company", "Security"),
        ("GICS Sector", "Sector"),
        Market.XNAS,
    ),
    "ftse100": UniverseSpec(
        "ftse100",
        "https://en.wikipedia.org/wiki/FTSE_100_Index",
        (95, 105),
        ("Ticker", "EPIC", "Symbol"),
        ("Company",),
        ("FTSE Industry Classification Benchmark sector", "Sector"),
        Market.XLON,
    ),
    "ftse250": UniverseSpec(
        "ftse250",
        "https://en.wikipedia.org/wiki/FTSE_250_Index",
        (240, 260),
        ("Ticker", "EPIC", "Symbol"),
        ("Company",),
        ("FTSE Industry Classification Benchmark sector", "Sector"),
        Market.XLON,
    ),
}


@dataclass
class ConstituentSnapshot:
    universe: str
    as_of: date
    members: pd.DataFrame  # symbol, name, sector, yahoo, market
    source_url: str
    html_sha256: str
    changes: pd.DataFrame | None = None  # date, added, removed, reason (canonical symbols)
    notes: list[str] = field(default_factory=list)


class WikipediaSource(BaseHTTPAdapter):
    """Fetches Wikipedia pages through the shared HTTP seam (tests feed recorded HTML instead)."""

    name = "wikipedia"
    rate_limit = RateLimit(
        min_interval_s=1.0, note="Wikipedia asks for a descriptive User-Agent and modest rates"
    )

    def capabilities(self) -> frozenset[Capability]:
        return frozenset({Capability.CONSTITUENTS})

    def fetch(self, url: str) -> str:
        return self._http("GET", url).text


# ----------------------------------------------------------------------------- symbol normalisation


def normalise_us_symbol(raw: str) -> tuple[str, str]:
    """('BRK.B' -> ('BRK-B', 'BRK-B')): canonical and Yahoo symbols for a US share class."""
    sym = str(raw).strip().upper().replace(".", "-")
    return sym, sym


def normalise_uk_symbol(raw: str) -> tuple[str, str]:
    """('BT.A' -> ('BT-A', 'BT-A.L'), 'BP.' -> ('BP', 'BP.L')): canonical and Yahoo symbols, LSE."""
    sym = str(raw).strip().upper().rstrip(".").replace(".", "-")
    sym = re.sub(r"\s+", "", sym)
    return sym, f"{sym}.L"


# ----------------------------------------------------------------------------- parsing


def _flatten_columns(df: pd.DataFrame) -> list[str]:
    out = []
    for col in df.columns:
        if isinstance(col, tuple):
            parts = [str(c) for c in col if str(c) and not str(c).startswith("Unnamed")]
            name = " ".join(dict.fromkeys(parts))
        else:
            name = str(col)
        out.append(re.sub(r"\[.*?\]", "", name).strip())
    return out


def _pick(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    for cand in candidates:
        for c in columns:
            if cand.lower() in c.lower():
                return c
    return None


def parse_constituents(
    html: str, spec: UniverseSpec, expected_rows: tuple[int, int] | None = None
) -> tuple[pd.DataFrame, str]:
    """Find the constituents table, normalise symbols and check the shape. Returns (members, html sha256)."""
    sha = hashlib.sha256(html.encode("utf-8")).hexdigest()
    lo, hi = expected_rows or spec.expected_rows
    try:
        tables = pd.read_html(io.StringIO(html), flavor="lxml")
    except (ValueError, ImportError) as exc:
        raise UniverseError(f"{spec.name}: no HTML tables found ({exc})") from exc
    chosen = None
    for tbl in tables:
        cols = _flatten_columns(tbl)
        if _pick(cols, spec.symbol_cols) and _pick(cols, spec.name_cols) and lo <= len(tbl) <= hi:
            tbl = tbl.copy()
            tbl.columns = cols
            chosen = tbl
            break
    if chosen is None:
        shapes = [(len(t), _flatten_columns(t)[:4]) for t in tables]
        raise UniverseError(
            f"{spec.name}: constituents table not found or shape drifted (want {lo}-{hi} rows with "
            f"{spec.symbol_cols[0]!r} and {spec.name_cols[0]!r}); tables seen: {shapes}"
        )
    sym_col = _pick(list(chosen.columns), spec.symbol_cols)
    name_col = _pick(list(chosen.columns), spec.name_cols)
    sector_col = _pick(list(chosen.columns), spec.sector_cols)
    normalise = normalise_uk_symbol if spec.market is Market.XLON else normalise_us_symbol
    rows = []
    for raw_sym, name, sector in zip(
        chosen[sym_col],
        chosen[name_col],
        chosen[sector_col] if sector_col else [None] * len(chosen),
        strict=True,
    ):
        if pd.isna(raw_sym):
            continue
        canonical, yahoo = normalise(str(raw_sym))
        rows.append(
            {
                "symbol": canonical,
                "name": str(name).strip(),
                "sector": None if sector is None or pd.isna(sector) else str(sector).strip(),
                "yahoo": yahoo,
                "market": spec.market.value,
            }
        )
    members = pd.DataFrame(rows).drop_duplicates("symbol").reset_index(drop=True)
    if not lo <= len(members) <= hi:
        raise UniverseError(f"{spec.name}: {len(members)} members after normalisation, expected {lo}-{hi}")
    return members, sha


def parse_sp500_changes(html: str) -> pd.DataFrame:
    """The S&P 500 'selected changes' table -> columns date, added, removed, reason (canonical symbols)."""
    try:
        tables = pd.read_html(io.StringIO(html), flavor="lxml")
    except (ValueError, ImportError) as exc:
        raise UniverseError(f"S&P 500 changes: no HTML tables found ({exc})") from exc
    for tbl in tables:
        cols = _flatten_columns(tbl)
        lower = [c.lower() for c in cols]
        has_added = any("added" in c and ("ticker" in c or "symbol" in c) for c in lower)
        has_removed = any("removed" in c and ("ticker" in c or "symbol" in c) for c in lower)
        date_col = next((c for c in cols if c.lower().startswith("date")), None)
        if has_added and has_removed and date_col:
            tbl = tbl.copy()
            tbl.columns = cols
            added_col = next(
                c for c in cols if "added" in c.lower() and ("ticker" in c.lower() or "symbol" in c.lower())
            )
            removed_col = next(
                c for c in cols if "removed" in c.lower() and ("ticker" in c.lower() or "symbol" in c.lower())
            )
            reason_col = next((c for c in cols if c.lower().startswith("reason")), None)
            out = pd.DataFrame(
                {
                    "date": pd.to_datetime(tbl[date_col], errors="coerce", format="mixed"),
                    "added": tbl[added_col].map(lambda v: None if pd.isna(v) else normalise_us_symbol(v)[0]),
                    "removed": tbl[removed_col].map(
                        lambda v: None if pd.isna(v) else normalise_us_symbol(v)[0]
                    ),
                    "reason": tbl[reason_col].astype(str) if reason_col else "",
                }
            )
            out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
            if out.empty:
                raise UniverseError("S&P 500 changes table parsed but no dated rows found")
            return out
    raise UniverseError("S&P 500 changes table not found: page layout may have drifted")


# ----------------------------------------------------------------------------- instruments


def members_to_instruments(
    members: pd.DataFrame, spec: UniverseSpec, catalog: InstrumentCatalog
) -> list[Instrument]:
    """Reuse catalogue instruments by Yahoo symbol; create the rest with conservative defaults."""
    out: list[Instrument] = []
    for row in members.itertuples(index=False):
        existing = catalog.by_vendor_symbol("yahoo", row.yahoo)
        if existing is not None:
            out.append(existing)
            continue
        if spec.market is Market.XLON:
            inst = Instrument(
                symbol=row.symbol,
                name=row.name,
                asset_class=AssetClass.EQUITY,
                market=Market.XLON,
                currency=Currency.GBP,
                price_scale=0.01,
                sector=row.sector,
                stamp_duty_class=StampDutyClass.UNKNOWN,  # taxed conservatively until classified
                identifiers={"yahoo": row.yahoo, "stooq": f"{row.symbol}.UK"},
                notes="from index constituents; GBX quote assumed, incorporation unknown",
            )
        else:
            inst = Instrument(
                symbol=row.symbol,
                name=row.name,
                asset_class=AssetClass.EQUITY,
                market=spec.market,
                currency=Currency.USD,
                sector=row.sector,
                stamp_duty_class=StampDutyClass.EXEMPT,
                country="US",
                identifiers={"yahoo": row.yahoo, "stooq": f"{row.symbol}.US"},
                notes="from index constituents; market is a calendar family, not a verified listing venue",
            )
        out.append(inst)
    return out


def load_snapshot(
    universe: str,
    source: WikipediaSource | None = None,
    *,
    html: str | None = None,
    changes_html: str | None = None,
    as_of: date | None = None,
    expected_rows: tuple[int, int] | None = None,
) -> ConstituentSnapshot:
    """Fetch (or parse supplied HTML for) a universe's current members, plus S&P changes when available."""
    if universe not in UNIVERSES:
        raise UniverseError(f"unknown universe {universe!r}; known: {sorted(UNIVERSES)}")
    spec = UNIVERSES[universe]
    if html is None:
        source = source or WikipediaSource()
        html = source.fetch(spec.url)
    members, sha = parse_constituents(html, spec, expected_rows)
    snap = ConstituentSnapshot(universe, as_of or today(), members, spec.url, sha)
    if spec.has_changes:
        try:
            snap.changes = parse_sp500_changes(changes_html or html)
        except UniverseError as exc:
            snap.notes.append(str(exc))
    else:
        snap.notes.append(
            f"{universe}: no changes table on the source page; "
            "history is current_only until snapshots accumulate"
        )
    return snap
