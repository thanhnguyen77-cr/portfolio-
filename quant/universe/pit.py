"""Point-in-time universe membership.

Membership is a set of half-open intervals per instrument: [effective_from, effective_to] with an open
end (NULL) for current members. Three ways in, in increasing trustworthiness:
  * `add_snapshot`: record today's members; repeated over time this builds true history forward
    (`current_only` until a change has actually been observed);
  * `reconstruct_from_changes`: walk an index's add/remove table backwards from a current snapshot
    (`reconstructed`: approximate, incomplete early on, tickers may have been reused);
  * `import_intervals`: licensed or hand-checked history (`pit_complete` only when the caller says so).
`coverage()` reports the share of historical members that actually have price data, so survivorship
holes are measured instead of assumed away.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from quant.core.errors import UniverseError
from quant.core.time import now, to_date
from quant.data.store import DuckDBStore

QUALITIES = ("current_only", "reconstructed", "pit_complete")


@dataclass(frozen=True)
class SnapshotDelta:
    as_of: date
    added: tuple[str, ...]
    removed: tuple[str, ...]
    members: int


class PITUniverse:
    def __init__(self, db: DuckDBStore, name: str) -> None:
        self.db = db
        self.name = name

    # ------------------------------------------------------------------ meta
    def quality(self) -> str:
        df = self.db.query("SELECT quality FROM universe_meta WHERE universe = ?", [self.name])
        return str(df["quality"].iloc[0]) if not df.empty else "current_only"

    def history_start(self) -> date | None:
        df = self.db.query("SELECT history_start FROM universe_meta WHERE universe = ?", [self.name])
        if df.empty or pd.isna(df["history_start"].iloc[0]):
            return None
        return pd.Timestamp(df["history_start"].iloc[0]).date()

    def set_quality(self, quality: str, source: str | None = None, history_start: date | None = None) -> None:
        if quality not in QUALITIES:
            raise UniverseError(f"quality must be one of {QUALITIES}")
        self.db.execute(
            "INSERT OR REPLACE INTO universe_meta VALUES (?, ?, ?, ?, ?)",
            [self.name, quality, source, history_start, now()],
        )

    # ------------------------------------------------------------------ reads
    def intervals(self) -> pd.DataFrame:
        df = self.db.query(
            "SELECT instrument_id, effective_from, effective_to, source, recorded_at "
            "FROM universe_membership "
            "WHERE universe = ? ORDER BY instrument_id, effective_from",
            [self.name],
        )
        if not df.empty:
            df["effective_from"] = pd.to_datetime(df["effective_from"]).dt.date
            df["effective_to"] = [None if pd.isna(v) else pd.Timestamp(v).date() for v in df["effective_to"]]
        return df

    def snapshots(self) -> pd.DataFrame:
        return self.db.query(
            "SELECT as_of, count(*) AS members, min(source) AS source FROM universe_snapshots "
            "WHERE universe = ? "
            "GROUP BY as_of ORDER BY as_of",
            [self.name],
        )

    def membership_at(self, d: date | str) -> list[str]:
        day = to_date(d)
        df = self.db.query(
            "SELECT instrument_id FROM universe_membership WHERE universe = ? AND effective_from <= ? "
            "AND (effective_to IS NULL OR effective_to >= ?) ORDER BY instrument_id",
            [self.name, day, day],
        )
        return df["instrument_id"].tolist()

    def members_over(self, start: date | str, end: date | str) -> list[str]:
        """Every instrument that was a member at any point in [start, end]."""
        s, e = to_date(start), to_date(end)
        df = self.db.query(
            "SELECT DISTINCT instrument_id FROM universe_membership "
            "WHERE universe = ? AND effective_from <= ? "
            "AND (effective_to IS NULL OR effective_to >= ?) ORDER BY instrument_id",
            [self.name, e, s],
        )
        return df["instrument_id"].tolist()

    def coverage(self, start: date | str, end: date | str, with_data: Collection[str]) -> float:
        """Share of members over [start, end] that have price data (1.0 = no survivorship hole visible)."""
        members = self.members_over(start, end)
        if not members:
            return 0.0
        have = set(with_data)
        return sum(1 for m in members if m in have) / len(members)

    # ------------------------------------------------------------------ writes
    def add_snapshot(self, as_of: date | str, instrument_ids: Iterable[str], source: str) -> SnapshotDelta:
        """Record the members observed on `as_of`; open/close intervals accordingly (append-only in time)."""
        day = to_date(as_of)
        ids = sorted(set(instrument_ids))
        if not ids:
            raise UniverseError("a snapshot needs at least one member")
        last = self.db.query("SELECT max(as_of) AS d FROM universe_snapshots WHERE universe = ?", [self.name])
        last_day = last["d"].iloc[0]
        if last_day is not None and not pd.isna(last_day) and pd.Timestamp(last_day).date() > day:
            raise UniverseError(
                f"snapshot {day} is older than the latest recorded snapshot {pd.Timestamp(last_day).date()}"
            )
        open_df = self.db.query(
            "SELECT instrument_id, effective_from FROM universe_membership "
            "WHERE universe = ? AND effective_to IS NULL",
            [self.name],
        )
        currently_open = set(open_df["instrument_id"].tolist())
        to_close = sorted(currently_open - set(ids))
        to_open = sorted(set(ids) - currently_open)
        ts = now()
        with self.db.write() as con:
            for iid in to_close:
                con.execute(
                    "UPDATE universe_membership SET effective_to = ? "
                    "WHERE universe = ? AND instrument_id = ? "
                    "AND effective_to IS NULL",
                    [day - timedelta(days=1), self.name, iid],
                )
            if to_open:
                con.executemany(
                    "INSERT OR REPLACE INTO universe_membership VALUES (?, ?, ?, NULL, ?, ?)",
                    [(self.name, iid, day, source, ts) for iid in to_open],
                )
            con.executemany(
                "INSERT OR REPLACE INTO universe_snapshots VALUES (?, ?, ?, ?, ?)",
                [(self.name, day, iid, source, ts) for iid in ids],
            )
        if (
            self.db.query("SELECT count(*) AS n FROM universe_meta WHERE universe = ?", [self.name])[
                "n"
            ].iloc[0]
            == 0
        ):
            self.set_quality("current_only", source, history_start=day)
        return SnapshotDelta(day, tuple(to_open), tuple(to_close), len(ids))

    def reconstruct_from_changes(
        self,
        current_ids: Iterable[str],
        as_of: date | str,
        changes: pd.DataFrame,
        *,
        source: str = "index_changes",
        history_start: date | None = None,
    ) -> int:
        """Rebuild intervals by walking `changes` (columns date, added_id, removed_id) back from a snapshot.

        Adjacent intervals of the same instrument are merged (a ticker rename mapped to one id becomes a
        single unbroken membership). Replaces this universe's intervals and marks it `reconstructed`.
        """
        day = to_date(as_of)
        members = set(current_ids)
        if not members:
            raise UniverseError("reconstruction needs the current members")
        ch = changes.copy()
        ch["date"] = pd.to_datetime(ch["date"]).dt.date
        ch = ch[ch["date"] <= day].sort_values("date", ascending=False)
        starts: dict[str, date] = {}
        closed: list[tuple[str, date, date]] = []  # (id, from, to) with from unknown at first -> filled later
        pending_closed: list[tuple[str, date]] = []  # (id, effective_to) whose start is still unknown
        for d, added, removed in ch[["date", "added_id", "removed_id"]].itertuples(index=False):
            if isinstance(added, str) and added:
                if added in members:
                    starts[added] = d
                    members.discard(added)
            if isinstance(removed, str) and removed:
                if removed not in members:
                    pending_closed.append((removed, d - timedelta(days=1)))
                    members.add(removed)
        earliest = history_start or ((ch["date"].min() - timedelta(days=1)) if not ch.empty else day)
        intervals: list[tuple[str, date, date | None]] = []
        removed_ids = {iid for iid, _ in pending_closed}
        for iid in members - removed_ids - set(starts):  # members at the start of history, never touched
            intervals.append((iid, earliest, None))
        for iid, start in starts.items():
            intervals.append((iid, start, None))
        # closed intervals start at the last recorded add before the removal, else at `earliest`
        adds_by_id: dict[str, list[date]] = {}
        for d, added, _ in ch[["date", "added_id", "removed_id"]].itertuples(index=False):
            if isinstance(added, str) and added:
                adds_by_id.setdefault(added, []).append(d)
        for iid, end in pending_closed:
            prior_adds = sorted(a for a in adds_by_id.get(iid, []) if a <= end)
            start = prior_adds[-1] if prior_adds else earliest
            closed.append((iid, start, end))
        # open intervals that were also re-added later need their earlier closed stints kept; merge adjacency
        rows = intervals + [(iid, s, e) for iid, s, e in closed]
        merged = _merge_adjacent(rows)
        ts = now()
        with self.db.write() as con:
            con.execute("DELETE FROM universe_membership WHERE universe = ?", [self.name])
            con.executemany(
                "INSERT INTO universe_membership VALUES (?, ?, ?, ?, ?, ?)",
                [(self.name, iid, s, e, source, ts) for iid, s, e in merged],
            )
        self.set_quality("reconstructed", source, history_start=earliest)
        return len(merged)

    def import_intervals(self, df: pd.DataFrame, source: str, *, complete: bool = False) -> int:
        """Load intervals (instrument_id, effective_from, effective_to) from licensed or checked data."""
        required = {"instrument_id", "effective_from"}
        if not required <= set(df.columns):
            raise UniverseError(f"import needs columns {sorted(required)}")
        rows = []
        for r in df.itertuples(index=False):
            end = getattr(r, "effective_to", None)
            end_d = None if end is None or pd.isna(end) else to_date(end)
            rows.append((self.name, str(r.instrument_id), to_date(r.effective_from), end_d, source, now()))
        merged = _merge_adjacent([(iid, s, e) for _, iid, s, e, _, _ in rows])
        with self.db.write() as con:
            con.execute("DELETE FROM universe_membership WHERE universe = ?", [self.name])
            con.executemany(
                "INSERT INTO universe_membership VALUES (?, ?, ?, ?, ?, ?)",
                [(self.name, iid, s, e, source, now()) for iid, s, e in merged],
            )
        starts = [s for _, s, _ in merged]
        self.set_quality(
            "pit_complete" if complete else "reconstructed",
            source,
            history_start=min(starts) if starts else None,
        )
        return len(merged)


def _merge_adjacent(rows: list[tuple[str, date, date | None]]) -> list[tuple[str, date, date | None]]:
    """Merge intervals of the same id that touch or overlap; open intervals absorb everything after them."""
    by_id: dict[str, list[tuple[date, date | None]]] = {}
    for iid, s, e in rows:
        by_id.setdefault(iid, []).append((s, e))
    out: list[tuple[str, date, date | None]] = []
    for iid, ivs in by_id.items():
        ivs.sort(key=lambda x: x[0])
        cur_s, cur_e = ivs[0]
        for s, e in ivs[1:]:
            if cur_e is None or s <= cur_e + timedelta(days=1):
                if cur_e is not None:
                    cur_e = None if e is None else max(cur_e, e)
            else:
                out.append((iid, cur_s, cur_e))
                cur_s, cur_e = s, e
        out.append((iid, cur_s, cur_e))
    return sorted(out, key=lambda x: (x[0], x[1]))
