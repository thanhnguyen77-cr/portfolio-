"""UK and US yield curves and the cross-asset regime board.

Run:  python examples/gilt_curve_regime.py   (FRED needs no key; BoE/ONS ids marked unverified may fail)
"""

from __future__ import annotations

from datetime import date

from quant.config.settings import load_settings
from quant.data.service import DataService
from quant.macro import curve_snapshot, inversion_flags, slope

START, END = date(2015, 1, 1), date(2024, 6, 28)


def main() -> None:
    svc = DataService(load_settings())
    us = {t: svc.macro_series(f"us_{t}", START, END) for t in ("2y", "5y", "10y", "30y")}
    print(curve_snapshot(us).to_string(index=False))
    flags = inversion_flags(slope(us["10y"], us["2y"]))
    inverted_now = bool(flags["inverted"].iloc[-1])
    n_inverted = int(flags["sessions_inverted"].iloc[-1])
    print(f"US 10y-2y inverted now: {inverted_now}; sessions inverted: {n_inverted}")
    try:
        uk = {t: svc.macro_series(f"uk_{t}", START, END) for t in ("5y", "10y", "20y")}
        print(curve_snapshot(uk).to_string(index=False))
    except Exception as exc:  # noqa: BLE001 - the UK ids are flagged unverified in the catalogue
        print(f"UK curve unavailable: {exc}")
    for region in ("us", "uk"):
        board = svc.macro_dashboard(region, end=END, refresh=None)
        print(f"\n{region.upper()} quadrant: {board['quadrant']}  signals: {board['risk_signals']}")
        print(board["table"].to_string(index=False))
        for n in board["notes"]:
            print(f"  note: {n}")


if __name__ == "__main__":
    main()
