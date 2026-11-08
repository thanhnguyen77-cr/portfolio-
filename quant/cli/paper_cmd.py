"""`quant paper ...`: paper-trading accounts, the daily job, manual orders, reconciliation and blotters.

Paper only. `quant paper live` exists to show that the live gate refuses, nothing more.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import typer

from quant.cli._common import ConfigOption, console, frame_table, settings_from
from quant.cli.backtest_cmd import _load_strategy_class, parse_params

paper_app = typer.Typer(
    no_args_is_help=True, help="PAPER trading: sim broker by default, Alpaca paper optional."
)

AccountArg = Annotated[str, typer.Argument(help="Paper account id.")]
DateOpt = Annotated[str | None, typer.Option("--date", "-d", help="Session date (default: latest complete).")]
NoRefreshOpt = Annotated[bool, typer.Option("--no-refresh", help="Never fetch; cached bars only.")]
RuleOpt = Annotated[list[str] | None, typer.Option("--rule", help="Pre-trade rule override key=value.")]
BrokerOpt = Annotated[str | None, typer.Option("--broker", help="sim | alpaca (default from config).")]


def _service(config_file):  # noqa: ANN001
    from quant.data.service import DataService

    return DataService(settings_from(config_file))


def _ledger(svc):  # noqa: ANN001,ANN202
    from quant.paper import Ledger

    return Ledger(svc.db)


def _rules(items: list[str] | None):  # noqa: ANN202
    from quant.paper import PreTradeRules

    overrides: dict[str, Any] = {}
    for k, v in parse_params(items).items():
        if k in {"blocklist", "blocked_regimes", "caution_regimes"}:
            overrides[k] = {s.strip() for s in str(v).split("|") if s.strip()}
        else:
            overrides[k] = v
    return PreTradeRules(**overrides)


def _broker(svc, ledger, account_id: str, instruments: list[str], broker_name: str | None):  # noqa: ANN001,ANN202
    from quant.backtest.costs import CostModel
    from quant.paper import AlpacaBroker, SimBroker

    state = ledger.load(account_id)
    name = broker_name or state.broker
    ids = list(dict.fromkeys(list(state.config.get("instruments", [])) + instruments))
    insts = {svc.resolve(i).instrument_id: svc.resolve(i) for i in ids}
    if ids != state.config.get("instruments", []):
        state.config["instruments"] = list(insts)
        ledger.save(state)
    if name == "alpaca":
        s = svc.settings
        key = s.alpaca_api_key.get_secret_value() if s.alpaca_api_key else None
        sec = s.alpaca_secret_key.get_secret_value() if s.alpaca_secret_key else None
        return state, AlpacaBroker(key, sec, paper=True, instruments=insts, account_id=account_id)
    return state, SimBroker(
        state,
        insts,
        cost_model=CostModel(),
        participation_cap=float(state.config.get("participation_cap", 0.10)),
        fractional_shares=bool(state.config.get("fractional_shares", False)),
        allow_short=bool(state.config.get("allow_short", False)),
        cash_policy=str(state.config.get("cash_policy", svc.settings.trading.cash_policy)),
        ledger=ledger,
    )


def _refs(instruments: str | None) -> list[str]:
    return [r.strip() for r in (instruments or "").split(",") if r.strip()]


@paper_app.command("init")
def init(
    account: AccountArg,
    cash: Annotated[float, typer.Option("--cash")] = 1_000_000.0,
    base: Annotated[str | None, typer.Option("--base", help="Base currency (default from config).")] = None,
    as_of: Annotated[
        str | None, typer.Option("--as-of", help="Last session already 'seen' (default: latest).")
    ] = None,
    broker: BrokerOpt = None,
    allow_short: Annotated[bool, typer.Option("--allow-short")] = False,
    config_file: ConfigOption = None,
) -> None:
    """Create a paper account (sim broker by default)."""
    from quant.core.calendars import latest_complete_session

    svc = _service(config_file)
    ledger = _ledger(svc)
    base_ccy = base or svc.settings.trading.base_currency
    name = broker or svc.settings.trading.default_broker
    start = date.fromisoformat(as_of) if as_of else latest_complete_session("XNYS")
    ledger.create_account(
        account,
        broker=name,
        base_currency=base_ccy,
        initial_cash=cash,
        as_of=start,
        config={"allow_short": allow_short, "instruments": []},
    )
    console.print(
        f"created paper account [bold]{account}[/bold] ({name}, {cash:,.2f} {base_ccy}, as of {start})"
    )


@paper_app.command("accounts")
def accounts(config_file: ConfigOption = None) -> None:
    """List paper accounts."""
    svc = _service(config_file)
    df = _ledger(svc).list_accounts()
    if df.empty:
        console.print("no paper accounts; run `quant paper init NAME`")
        return
    console.print(frame_table(df, title="paper accounts"))


@paper_app.command("run")
def run(
    account: AccountArg,
    strategy: Annotated[str, typer.Option("--strategy", "-S", help="fixed_weights | trend_following | ...")],
    instruments: Annotated[str, typer.Option("--instruments", "-i", help="Comma-separated refs.")],
    weights: Annotated[str | None, typer.Option("--weights", help="fixed_weights: ref=w,...")] = None,
    params: Annotated[list[str] | None, typer.Option("--param", "-p", help="Strategy key=value.")] = None,
    session: DateOpt = None,
    rule: RuleOpt = None,
    html: Annotated[Path | None, typer.Option("--html", help="Write a blotter afterwards.")] = None,
    broker: BrokerOpt = None,
    no_refresh: NoRefreshOpt = False,
    config_file: ConfigOption = None,
) -> None:
    """Run one paper session: replay fills, let the strategy decide, gate and submit orders, reconcile."""
    from quant.paper import PaperRunner, build_blotter

    svc = _service(config_file)
    ledger = _ledger(svc)
    refs = [svc.resolve(r).instrument_id for r in _refs(instruments)]
    if not refs:
        raise typer.BadParameter("give at least one instrument")
    state, brk = _broker(svc, ledger, account, refs, broker)
    kwargs = parse_params(params)
    if weights:
        kwargs["weights"] = {
            svc.resolve(k).instrument_id: float(v) for k, v in (p.split("=", 1) for p in weights.split(","))
        }
    strat = _load_strategy_class(strategy)(refs, **kwargs)
    runner = PaperRunner(svc, brk, strat, _rules(rule), refresh=False if no_refresh else None)
    report = runner.run_session(session)
    console.print(report.summary())
    if report.fills:
        console.print(frame_table(pd.DataFrame([f.as_dict() for f in report.fills]), title="fills this run"))
    if report.submitted:
        console.print(frame_table(pd.DataFrame([o.as_dict() for o in report.submitted]), title="submitted"))
    if report.blocked:
        console.print(frame_table(report.blocked_table(), title="blocked by pre-trade rules"))
    if report.broker_rejected:
        console.print(
            frame_table(pd.DataFrame([o.as_dict() for o in report.broker_rejected]), title="rejected")
        )
    if html:
        build_blotter(
            state if hasattr(brk, "state") else state, brk, reconciliation=report.reconciliation,
            pretrade_rejections=report.blocked_table() if report.blocked else None, path=html,
        )  # fmt: skip
        console.print(f"wrote {html}")
    if report.reconciliation is not None and not report.reconciliation.ok:
        raise typer.Exit(code=1)


@paper_app.command("status")
def status(account: AccountArg, broker: BrokerOpt = None, config_file: ConfigOption = None) -> None:
    """Account snapshot, positions and open orders."""
    svc = _service(config_file)
    state, brk = _broker(svc, _ledger(svc), account, [], broker)
    acct = brk.account().as_dict()
    rows = [{"field": k, "value": str(v)} for k, v in acct.items()]
    console.print(frame_table(pd.DataFrame(rows), title=f"account {account} (PAPER)"))
    pos = pd.DataFrame([p.as_dict() for p in brk.positions()])
    console.print(frame_table(pos, title="positions") if not pos.empty else "no positions")
    oo = pd.DataFrame([o.as_dict() for o in brk.open_orders()])
    console.print(frame_table(oo, title="open orders") if not oo.empty else "no open orders")


@paper_app.command("orders")
def orders(
    account: AccountArg,
    all_orders: Annotated[bool, typer.Option("--all", help="Include closed orders.")] = False,
    config_file: ConfigOption = None,
) -> None:
    """Open (or all) orders."""
    svc = _service(config_file)
    _, brk = _broker(svc, _ledger(svc), account, [], None)
    rows = brk.orders() if all_orders else brk.open_orders()
    df = pd.DataFrame([o.as_dict() for o in rows])
    console.print(frame_table(df, title="orders", max_rows=200) if not df.empty else "no orders")


@paper_app.command("fills")
def fills(account: AccountArg, config_file: ConfigOption = None) -> None:
    """Fill history."""
    svc = _service(config_file)
    _, brk = _broker(svc, _ledger(svc), account, [], None)
    df = pd.DataFrame([f.as_dict() for f in brk.fills()])
    console.print(frame_table(df, title="fills", max_rows=200) if not df.empty else "no fills")


@paper_app.command("submit")
def submit(
    account: AccountArg,
    instrument: Annotated[str, typer.Option("--instrument", "-i")],
    side: Annotated[str, typer.Option("--side", help="buy | sell")],
    qty: Annotated[float, typer.Option("--qty")],
    order_type: Annotated[str, typer.Option("--type", help="market | limit | stop | stop_limit")] = "market",
    limit_price: Annotated[float | None, typer.Option("--limit")] = None,
    stop_price: Annotated[float | None, typer.Option("--stop")] = None,
    tif: Annotated[str, typer.Option("--tif", help="day | gtc")] = "day",
    stop_loss: Annotated[float | None, typer.Option("--stop-loss", help="For the risk checks only.")] = None,
    take_profit: Annotated[
        float | None, typer.Option("--take-profit", help="For the risk checks only.")
    ] = None,
    session: DateOpt = None,
    rule: RuleOpt = None,
    broker: BrokerOpt = None,
    no_refresh: NoRefreshOpt = False,
    config_file: ConfigOption = None,
) -> None:
    """Hand-written order through the pre-trade gate (and, if it passes, to the paper broker)."""
    from quant.paper import OrderRequest, PaperRunner, submit_manual
    from quant.strategies import FixedWeights

    svc = _service(config_file)
    ledger = _ledger(svc)
    iid = svc.resolve(instrument).instrument_id
    state, brk = _broker(svc, ledger, account, [iid], broker)
    runner = PaperRunner(
        svc, brk, FixedWeights([iid], rebalance="never"), _rules(rule), refresh=False if no_refresh else None
    )
    req = OrderRequest(
        iid, side, qty, order_type, limit_price=limit_price, stop_price=stop_price, tif=tif, tag="manual"
    )
    report, order, new_fills = submit_manual(
        runner, req, session_date=session, stop_loss=stop_loss, take_profit=take_profit
    )
    console.print(frame_table(report.table(), title="pre-trade checks", max_rows=100))
    if new_fills:
        console.print(f"{len(new_fills)} resting order(s) filled while catching up")
    if order is None:
        console.print(f"[red]{report.summary()}[/red]")
        raise typer.Exit(code=1)
    console.print(
        f"order {order.order_id}: {order.status.value}"
        + (f" ({'; '.join(order.notes)})" if order.notes else "")
    )
    if order.status.value == "rejected":
        raise typer.Exit(code=1)


@paper_app.command("cancel")
def cancel(
    account: AccountArg, order_id: Annotated[str, typer.Argument()], config_file: ConfigOption = None
) -> None:
    """Cancel a resting order."""
    svc = _service(config_file)
    _, brk = _broker(svc, _ledger(svc), account, [], None)
    o = brk.cancel(order_id)
    console.print(f"order {o.order_id}: {o.status.value}")


@paper_app.command("reconcile")
def reconcile_cmd(account: AccountArg, config_file: ConfigOption = None) -> None:
    """Check ledger invariants and compare with the broker's view."""
    from quant.paper import reconcile

    svc = _service(config_file)
    state, brk = _broker(svc, _ledger(svc), account, [], None)
    rep = reconcile(state, brk)
    console.print(rep.summary())
    if not rep.ok:
        raise typer.Exit(code=1)


@paper_app.command("blotter")
def blotter(
    account: AccountArg,
    html: Annotated[Path, typer.Option("--html")],
    inline_plotly: Annotated[bool, typer.Option("--inline-plotly")] = False,
    config_file: ConfigOption = None,
) -> None:
    """Write the account blotter (HTML)."""
    from quant.paper import build_blotter, reconcile

    svc = _service(config_file)
    state, brk = _broker(svc, _ledger(svc), account, [], None)
    build_blotter(state, brk, reconciliation=reconcile(state, brk), path=html, inline_plotly=inline_plotly)
    console.print(f"wrote {html}")


@paper_app.command("live")
def live(config_file: ConfigOption = None) -> None:
    """Show what the live-trading gate says (it refuses: this version has no live broker)."""
    from quant.config.live_gate import require_live_confirmation
    from quant.core.errors import LiveTradingDisabled

    try:
        require_live_confirmation(settings_from(config_file), confirm=None)
    except LiveTradingDisabled as exc:
        console.print(f"[yellow]live trading disabled: {exc}[/yellow]")
        raise typer.Exit(code=2) from None
    except NotImplementedError as exc:  # pragma: no cover - only after every gate passes
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from None
