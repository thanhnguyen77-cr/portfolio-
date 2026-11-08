"""Paper trading: ledger persistence and invariants, SimBroker fills pinned to the engine, broker
validation and multi-currency behaviour, pre-trade rules, Alpaca paper-only adapter, the runner."""

from __future__ import annotations

import copy
from datetime import date

import numpy as np
import pandas as pd
import pytest
from conftest import ScheduleStrategy, make_bars

from quant.backtest.costs import CostModel
from quant.backtest.engine import Backtest, BacktestConfig
from quant.backtest.fills import BarView
from quant.core.calendars import union_sessions
from quant.core.errors import AdapterNotConfigured, BrokerError, LiveTradingDisabled
from quant.core.time import now
from quant.data.fx import FXRates
from quant.data.service import DataService
from quant.data.store import DuckDBStore, ParquetStore
from quant.paper import (
    AccountState,
    AlpacaBroker,
    Broker,
    Ledger,
    OrderRequest,
    PaperRunner,
    PreTradeRules,
    SimBroker,
    build_blotter,
    check_order,
    ledger_invariants,
    reconcile,
    submit_manual,
)
from quant.paper.broker import AccountSnapshot, BrokerOrder, Position
from quant.strategies import FixedWeights

S, E = date(2024, 1, 2), date(2024, 3, 28)


def _views(bars: pd.DataFrame, ts: pd.Timestamp) -> dict[str, BarView]:
    day = bars[bars["session_date"] == ts]
    return {
        str(r.instrument_id): BarView(
            float(r.open), float(r.high), float(r.low), float(r.close), float(r.volume)
        )
        for r in day.itertuples(index=False)
    }


def _state(cash: float = 1_000_000.0, base: str = "USD", as_of: date | None = None) -> AccountState:
    st = AccountState("acct", "sim", base, cash, now(), as_of=as_of, cash={base: cash})
    st.add_event("deposit", as_of, "initial deposit", amount=cash, currency=base)
    return st


# ---------------------------------------------------------------- SimBroker == engine
def test_sim_broker_fills_match_engine(catalog):
    ids = ["XNAS_AAPL", "XNYS_JPM"]
    bars = pd.concat(
        [make_bars(catalog[i], S, E, seed=k + 3, start_price=100.0 * (k + 1)) for k, i in enumerate(ids)]
    )
    closes = bars.pivot(index="session_date", columns="instrument_id", values="close")
    sessions = union_sessions(["XNAS", "XNYS"], S, E)
    c = lambda i, iid: float(closes.loc[sessions[i], iid])  # noqa: E731
    schedule = {
        0: [
            ("XNAS_AAPL", "buy", 500),
            ("XNYS_JPM", "buy", 300, {"type": "limit", "limit_price": c(0, "XNYS_JPM") * 1.05}),
        ],
        10: [
            ("XNAS_AAPL", "sell", 200),
            (
                "XNAS_AAPL",
                "buy",
                50,
                {"type": "limit", "limit_price": c(10, "XNAS_AAPL") * 0.5, "tif": "gtc"},
            ),
        ],
        20: [
            ("XNYS_JPM", "sell", 300, {"type": "stop", "stop_price": c(20, "XNYS_JPM") * 0.97, "tif": "gtc"})
        ],
        30: [
            (
                "XNAS_AAPL",
                "buy",
                100,
                {
                    "type": "stop_limit",
                    "stop_price": c(30, "XNAS_AAPL") * 1.01,
                    "limit_price": c(30, "XNAS_AAPL") * 1.03,
                },
            )
        ],
        40: [("XNAS_AAPL", "sell", 200)],  # within the long position (an oversized sell would be rejected)
    }
    cfg = BacktestConfig(
        start=S,
        end=E,
        initial_cash=1_000_000.0,
        cost_model=CostModel(),
        participation_cap=0.10,
        fractional_shares=False,
        min_notional=0.0,
    )
    # ScheduleStrategy pops keys from the option dicts it is given, so the engine gets its own deep copy
    engine = Backtest(cfg, ScheduleStrategy(ids, copy.deepcopy(schedule)), bars, catalog).run()

    broker = SimBroker(_state(), {i: catalog[i] for i in ids}, cost_model=CostModel(), participation_cap=0.10)
    for idx, ts in enumerate(sessions):
        broker.process_session(ts.date(), _views(bars, ts))
        if idx == len(sessions) - 1:
            break
        for iid, side, qty, *rest in schedule.get(idx, []):
            kw = dict(rest[0]) if rest else {}
            kind = kw.pop("type", "market")
            o = broker.submit(OrderRequest(iid, side, qty, kind, **kw))
            assert o.status.value != "rejected", o.notes

    eng = engine.fills.sort_values(["session_date", "instrument_id", "qty"]).reset_index(drop=True)
    sim = pd.DataFrame([f.as_dict() for f in broker.fills()])
    sim["session_date"] = pd.to_datetime(sim["session_date"])
    sim = sim.sort_values(["session_date", "instrument_id", "qty"]).reset_index(drop=True)
    assert len(eng) == len(sim) >= 4  # the stop orders may or may not trigger on this sample path
    for col in ("qty", "price", "commission", "spread", "slippage", "taxes"):
        np.testing.assert_allclose(
            sim[col].to_numpy(dtype=float), eng[col].to_numpy(dtype=float), rtol=1e-12, err_msg=col
        )
    assert list(sim["instrument_id"]) == list(eng["instrument_id"])
    eq_sim = np.array([r["equity"] for r in broker.state.equity])
    np.testing.assert_allclose(eq_sim, engine.equity["equity"].to_numpy(), rtol=1e-10)
    assert ledger_invariants(broker.state) == []
    assert sum(f.total_cost for f in broker.fills()) > 0  # the shared cost model really charged something


# ---------------------------------------------------------------- SimBroker lifecycle
def test_sim_broker_validation_and_order_lifecycle(catalog):
    ids = ["XNAS_AAPL", "XNYS_SPX"]
    bars = make_bars(catalog["XNAS_AAPL"], S, E, seed=1, start_price=100.0)
    sessions = [s.date() for s in union_sessions(["XNAS"], S, E)]
    broker = SimBroker(
        _state(cash=50_000_000.0),
        {i: catalog[i] for i in ids},
        cost_model=CostModel(),
        participation_cap=0.10,
    )
    with pytest.raises(BrokerError, match="unknown instrument"):
        broker.submit(OrderRequest("XLON_VOD", "buy", 10))
    noref = broker.submit(OrderRequest("XNAS_AAPL", "buy", 10))
    assert noref.status.value == "rejected" and "reference price" in noref.notes[0]
    broker.process_session(sessions[0], _views(bars, pd.Timestamp(sessions[0])))
    mark = broker.state.marks["XNAS_AAPL"]
    assert broker.account().equity == pytest.approx(50_000_000.0)
    too_big = broker.submit(OrderRequest("XNAS_AAPL", "buy", 1_000_000))
    assert too_big.status.value == "rejected" and "buying power" in too_big.notes[0]
    short = broker.submit(OrderRequest("XNAS_AAPL", "sell", 10))
    assert short.status.value == "rejected" and "long-only" in short.notes[0]
    frac = broker.submit(OrderRequest("XNAS_AAPL", "buy", 1.5))
    assert frac.status.value == "rejected" and "lot size" in frac.notes[0]
    index = broker.submit(OrderRequest("XNYS_SPX", "buy", 1))
    assert index.status.value == "rejected" and "not tradable" in index.notes[0]
    with pytest.raises(BrokerError, match="not after"):
        broker.process_session(sessions[0], {})

    buy = broker.submit(OrderRequest("XNAS_AAPL", "buy", 100))
    assert (
        buy.status.value == "open" and buy.eligible_from == sessions[1] and buy.created_session == sessions[0]
    )
    cancelled = broker.cancel(buy.order_id)
    assert cancelled.status.value == "cancelled"
    with pytest.raises(BrokerError, match="cannot be cancelled"):
        broker.cancel(buy.order_id)
    with pytest.raises(BrokerError, match="not found"):
        broker.cancel("nope")
    low_day = broker.submit(OrderRequest("XNAS_AAPL", "buy", 10, "limit", limit_price=mark * 0.5))
    low_gtc = broker.submit(OrderRequest("XNAS_AAPL", "buy", 10, "limit", limit_price=mark * 0.5, tif="gtc"))
    vol = float(bars.loc[bars["session_date"] == pd.Timestamp(sessions[1]), "volume"].iloc[0])
    big = broker.submit(OrderRequest("XNAS_AAPL", "buy", int(vol * 0.3)))  # 3x the participation cap
    assert big.status.value == "open"
    fills = broker.process_session(sessions[1], _views(bars, pd.Timestamp(sessions[1])))
    assert broker.state.orders[low_day.order_id].status.value == "expired"
    assert broker.state.orders[low_gtc.order_id].status.value == "open"
    assert len(fills) == 1 and fills[0].qty == pytest.approx(np.floor(vol * 0.10))
    assert broker.state.orders[big.order_id].status.value == "expired"  # day order: remainder expired
    assert "partial fill" in broker.state.orders[big.order_id].notes[-1]
    pos = broker.positions()
    assert len(pos) == 1 and pos[0].qty == fills[0].qty and pos[0].avg_price == pytest.approx(fills[0].price)
    acct = broker.account()
    assert acct.equity == pytest.approx(acct.cash_base + acct.positions_value) and acct.n_open_orders == 1
    assert acct.last_equity == pytest.approx(50_000_000.0)
    assert ledger_invariants(broker.state) == []
    assert isinstance(broker, Broker)


def test_sim_broker_multi_currency_cash_and_invariants(catalog):
    ids = ["XNAS_AAPL", "XLON_VOD"]
    bars = pd.concat(
        [
            make_bars(catalog["XNAS_AAPL"], S, E, seed=4),
            make_bars(catalog["XLON_VOD"], S, E, seed=5, start_price=0.7),
        ]
    )
    fx_bars = make_bars(catalog["FX_GBPUSD"], S, E, seed=6, start_price=1.27, vol=0.003, drift=0.0)
    fx = FXRates.from_bars(fx_bars, catalog)
    broker = SimBroker(
        _state(), {i: catalog[i] for i in ids}, cost_model=CostModel(), fx=fx, participation_cap=None
    )
    sessions = union_sessions(["XNAS", "XLON"], S, E)
    broker.process_session(sessions[0].date(), _views(bars, sessions[0]))
    o = broker.submit(OrderRequest("XLON_VOD", "buy", 10_000))
    assert o.status.value == "open"
    for ts in sessions[1:6]:
        broker.process_session(ts.date(), _views(bars, ts))
        acct = broker.account()
        manual = (
            sum(amt * fx.latest(ccy, "USD", ts.date()) for ccy, amt in acct.cash.items())
            + acct.positions_value
        )
        assert acct.equity == pytest.approx(manual, rel=1e-12)
        assert ledger_invariants(broker.state) == []
    f = broker.fills()[0]
    assert f.currency == "GBP" and f.taxes > 0  # UK stamp duty on a uk_ordinary buy
    assert broker.state.cash.get("GBP", 0.0) == pytest.approx(
        0.0, abs=1e-6
    )  # converted exactly what the buy needed
    kinds = {e["type"] for e in broker.state.events}
    assert {"fx_out", "fx_in", "fill", "cost"} <= kinds
    sell = broker.submit(OrderRequest("XLON_VOD", "sell", 10_000))
    assert sell.status.value == "open"
    for ts in sessions[6:8]:
        broker.process_session(ts.date(), _views(bars, ts))
    assert broker.state.cash.get("GBP", 0.0) == pytest.approx(0.0, abs=1e-6)  # proceeds swept to base
    assert not broker.positions() and ledger_invariants(broker.state) == []


# ---------------------------------------------------------------- ledger persistence
def test_ledger_round_trip_and_invariant_detection(tmp_path, catalog):
    db = DuckDBStore(tmp_path / "q.duckdb")
    ledger = Ledger(db)
    assert {"paper_accounts", "paper_fills", "paper_events"} <= set(db.tables())
    state = ledger.create_account(
        "demo", broker="sim", base_currency="USD", initial_cash=250_000.0, as_of=None
    )
    with pytest.raises(Exception, match="already exists"):
        ledger.create_account("demo", broker="sim", base_currency="USD", initial_cash=1.0)
    bars = make_bars(catalog["XNAS_AAPL"], S, E, seed=2)
    sessions = union_sessions(["XNAS"], S, E)
    broker = SimBroker(state, {"XNAS_AAPL": catalog["XNAS_AAPL"]}, cost_model=CostModel(), ledger=ledger)
    broker.process_session(sessions[0].date(), _views(bars, sessions[0]))
    broker.submit(OrderRequest("XNAS_AAPL", "buy", 100, tag="t1"))
    broker.submit(OrderRequest("XNAS_AAPL", "buy", 5, "limit", limit_price=1.0, tif="gtc"))
    broker.process_session(sessions[1].date(), _views(bars, sessions[1]))
    loaded = ledger.load("demo")
    assert loaded.cash == pytest.approx(state.cash) and loaded.as_of == state.as_of
    assert {k: (p.qty, p.avg_price) for k, p in loaded.positions.items()} == {
        k: (p.qty, p.avg_price) for k, p in state.positions.items()
    }
    assert [o.as_dict() for o in loaded.orders.values()] == [o.as_dict() for o in state.orders.values()]
    assert [f.as_dict() for f in loaded.fills] == [f.as_dict() for f in state.fills]
    assert len(loaded.equity) == 2 and loaded.equity[-1]["equity"] == pytest.approx(
        state.equity[-1]["equity"]
    )
    assert len(loaded.events) == len(state.events) and loaded.marks == state.marks
    assert ledger_invariants(loaded) == [] and reconcile(loaded, SimBroker(loaded, broker.instruments)).ok
    assert ledger.list_accounts()["account_id"].tolist() == ["demo"]
    loaded.cash["USD"] += 1.0
    loaded.positions["XNAS_AAPL"].qty += 1.0
    issues = ledger_invariants(loaded)
    assert any(i.startswith("cash USD") for i in issues) and any(
        i.startswith("position XNAS_AAPL") for i in issues
    )
    ledger.delete_account("demo")
    assert not ledger.exists("demo")


# ---------------------------------------------------------------- pre-trade rules
def _acct(equity=100_000.0, cash=None, last=100_000.0) -> AccountSnapshot:
    cash = equity if cash is None else cash
    return AccountSnapshot(
        "a", "sim", date(2024, 6, 28), "USD", {"USD": cash}, cash, equity - cash, equity, last, 0, 0
    )


def test_pretrade_rules_each_trigger(catalog):
    inst = catalog["XNAS_AAPL"]
    rules = PreTradeRules()
    base = dict(instrument=inst, ref_price=100.0, adv=1_000_000.0, regime="risk_on", days_to_earnings=30)
    ok = check_order(OrderRequest("XNAS_AAPL", "buy", 50), rules, _acct(), [], [], **base)
    assert ok.passed and not ok.warnings, ok.summary()
    assert set(ok.table()["status"]) == {"pass"} and "PASS" in ok.summary()

    def fails(report, name):
        return name in {c.name for c in report.failures}

    assert fails(
        check_order(OrderRequest("XNAS_AAPL", "buy", 50), rules, _acct(), [], [], live=True, **base), "mode"
    )
    assert fails(
        check_order(OrderRequest("ZZZ", "buy", 1), rules, _acct(), [], [], **{**base, "instrument": None}),
        "instrument",
    )
    assert fails(
        check_order(
            OrderRequest("XNAS_AAPL", "buy", 50), PreTradeRules(blocklist={"AAPL"}), _acct(), [], [], **base
        ),
        "blocklist",
    )
    assert fails(check_order(OrderRequest("XNAS_AAPL", "sell", 50), rules, _acct(), [], [], **base), "side")
    held = [Position("XNAS_AAPL", 100, 90.0, "USD", 100.0, 10_000.0)]
    assert check_order(OrderRequest("XNAS_AAPL", "sell", 50), rules, _acct(), held, [], **base).passed
    assert fails(check_order(OrderRequest("XNAS_AAPL", "buy", 1.5), rules, _acct(), [], [], **base), "qty")
    assert fails(
        check_order(
            OrderRequest("XNAS_AAPL", "buy", 50),
            PreTradeRules(allow_market_orders=False),
            _acct(),
            [],
            [],
            **base,
        ),
        "order_type",
    )
    bad_sl = OrderRequest("XNAS_AAPL", "buy", 50, "stop_limit", limit_price=100.0, stop_price=101.0)
    assert fails(check_order(bad_sl, rules, _acct(), [], [], **base), "stop_limit_prices")
    assert fails(
        check_order(
            OrderRequest("XNAS_AAPL", "buy", 50), rules, _acct(), [], [], **{**base, "ref_price": 0.5}
        ),
        "min_price",
    )
    assert fails(
        check_order(
            OrderRequest("XNAS_AAPL", "buy", 50),
            PreTradeRules(require_protective_stop=True),
            _acct(),
            [],
            [],
            **base,
        ),
        "protective_stop",
    )
    assert fails(
        check_order(OrderRequest("XNAS_AAPL", "buy", 50), rules, _acct(), [], [], take_profit=99.0, **base),
        "take_profit",
    )
    assert fails(
        check_order(
            OrderRequest("XNAS_AAPL", "buy", 50),
            rules,
            _acct(),
            [],
            [],
            stop_loss=95.0,
            take_profit=104.0,
            **base,
        ),
        "reward_risk",
    )
    assert check_order(
        OrderRequest("XNAS_AAPL", "buy", 50),
        rules,
        _acct(),
        [],
        [],
        stop_loss=95.0,
        take_profit=110.0,
        **base,
    ).passed
    assert fails(
        check_order(OrderRequest("XNAS_AAPL", "buy", 200), rules, _acct(), [], [], **base), "max_position_pct"
    )
    assert fails(
        check_order(
            OrderRequest("XNAS_AAPL", "buy", 50), PreTradeRules(max_order_pct=1.0), _acct(), [], [], **base
        ),
        "max_order_pct",
    )
    assert fails(
        check_order(OrderRequest("XNAS_AAPL", "buy", 50), rules, _acct(), [], [], stop_loss=50.0, **base),
        "max_risk_per_trade",
    )
    assert fails(
        check_order(OrderRequest("XNAS_AAPL", "buy", 50), rules, _acct(cash=7_000.0), [], [], **base),
        "cash_reserve",
    )
    heavy = [Position("XNAS_MSFT", 950, 100.0, "USD", 100.0, 95_000.0)]
    assert fails(
        check_order(OrderRequest("XNAS_AAPL", "buy", 60), rules, _acct(cash=100_000.0), heavy, [], **base),
        "total_exposure",
    )
    assert fails(
        check_order(
            OrderRequest("XNAS_AAPL", "buy", 50),
            PreTradeRules(max_open_positions=1),
            _acct(),
            heavy,
            [],
            **base,
        ),
        "max_open_positions",
    )
    resting = [BrokerOrder("o1", "XNAS_AAPL", "buy", 10, eligible_from=date(2024, 7, 1))]
    assert fails(
        check_order(OrderRequest("XNAS_AAPL", "buy", 50), rules, _acct(), [], resting, **base), "no_duplicate"
    )
    assert fails(
        check_order(
            OrderRequest("XNAS_AAPL", "buy", 50),
            rules,
            _acct(equity=96_000.0, last=100_000.0),
            [],
            [],
            **base,
        ),
        "daily_loss_limit",
    )
    assert fails(
        check_order(OrderRequest("XNAS_AAPL", "buy", 50), rules, _acct(), [], [], **{**base, "adv": 100.0}),
        "participation",
    )
    assert fails(
        check_order(
            OrderRequest("XNAS_AAPL", "buy", 50), rules, _acct(), [], [], **{**base, "regime": "risk_off"}
        ),
        "regime",
    )
    caution = check_order(
        OrderRequest("XNAS_AAPL", "buy", 50), rules, _acct(), [], [], **{**base, "regime": "caution"}
    )
    assert caution.passed and "regime" in {c.name for c in caution.warnings}
    unknown = check_order(
        OrderRequest("XNAS_AAPL", "buy", 50),
        rules,
        _acct(last=None),
        [],
        [],
        **{**base, "regime": None, "adv": None},
    )
    assert unknown.passed and {"regime", "daily_loss_limit", "participation"} <= {
        c.name for c in unknown.warnings
    }
    assert fails(
        check_order(
            OrderRequest("XNAS_AAPL", "buy", 50), rules, _acct(), [], [], **{**base, "days_to_earnings": 1}
        ),
        "earnings",
    )
    assert (
        "BLOCK"
        in check_order(
            OrderRequest("XNAS_AAPL", "buy", 50), rules, _acct(), [], [], live=True, **base
        ).summary()
    )


# ---------------------------------------------------------------- Alpaca paper adapter
class _FakeAlpacaClient:
    def __init__(self, key, secret, paper):
        self.kwargs = {"key": key, "secret": secret, "paper": paper}
        self.submitted = []
        self.cancelled = []

    def get_account(self):
        return {
            "cash": "50000",
            "equity": "80000",
            "last_equity": "79000",
            "long_market_value": "30000",
            "short_market_value": "0",
            "account_number": "PA123",
        }

    def get_all_positions(self):
        return [
            {
                "symbol": "AAPL",
                "qty": "10",
                "side": "long",
                "avg_entry_price": "150",
                "current_price": "160",
                "market_value": "1600",
            }
        ]

    def get_orders(self, *a, **k):
        return [
            {
                "id": "o-1",
                "symbol": "AAPL",
                "side": "buy",
                "qty": "10",
                "type": "limit",
                "limit_price": "150",
                "time_in_force": "day",
                "status": "filled",
                "filled_qty": "10",
                "filled_avg_price": "149.5",
                "created_at": "2024-06-27T14:30:00Z",
                "client_order_id": "c1",
            },
            {
                "id": "o-2",
                "symbol": "MSFT",
                "side": "buy",
                "qty": "5",
                "type": "market",
                "time_in_force": "day",
                "status": "new",
                "filled_qty": "0",
                "created_at": "2024-06-28T14:30:00Z",
            },
        ]

    def submit_order(self, order_data):
        self.submitted.append(order_data)
        return {
            "id": "o-3",
            "symbol": order_data["symbol"],
            "side": order_data["side"],
            "qty": order_data["qty"],
            "type": order_data["type"],
            "time_in_force": order_data["time_in_force"],
            "status": "accepted",
            "filled_qty": "0",
            "limit_price": order_data.get("limit_price"),
        }

    def cancel_order_by_id(self, oid):
        self.cancelled.append(oid)

    def get_order_by_id(self, oid):
        return {
            "id": oid,
            "symbol": "MSFT",
            "side": "buy",
            "qty": "5",
            "type": "market",
            "time_in_force": "day",
            "status": "canceled",
            "filled_qty": "0",
        }


def test_alpaca_broker_is_paper_only_and_maps_fields(catalog):
    with pytest.raises(LiveTradingDisabled):
        AlpacaBroker("k", "s", paper=False)
    with pytest.raises(AdapterNotConfigured, match="QUANT_ALPACA"):
        AlpacaBroker(None, None, client_factory=_FakeAlpacaClient).account()
    insts = {i: catalog[i] for i in ("XNAS_AAPL", "XNAS_MSFT", "XLON_VOD")}
    broker = AlpacaBroker("key", "secret", instruments=insts, client_factory=_FakeAlpacaClient)
    assert isinstance(broker, Broker) and broker.paper
    acct = broker.account()
    assert broker._client.kwargs == {"key": "key", "secret": "secret", "paper": True}
    assert acct.equity == 80_000.0 and acct.cash == {"USD": 50_000.0} and acct.account_id == "PA123"
    assert acct.day_change_pct == pytest.approx((80_000 / 79_000 - 1) * 100)
    pos = broker.positions()
    assert pos[0].instrument_id == "XNAS_AAPL" and pos[0].qty == 10 and pos[0].market_value_base == 1600.0
    orders = broker.orders()
    assert [o.status.value for o in orders] == ["filled", "open"] and orders[0].instrument_id == "XNAS_AAPL"
    assert len(broker.open_orders()) == 1 and broker.fills()[0].price == 149.5
    o = broker.submit(OrderRequest("XNAS_MSFT", "buy", 5, "limit", limit_price=400.0, tif="gtc", tag="t"))
    sent = broker._client.submitted[0]
    assert (
        sent["symbol"] == "MSFT"
        and sent["side"] == "buy"
        and sent["type"] == "limit"
        and sent["time_in_force"] == "gtc"
    )
    assert o.order_id == "o-3" and o.status.value == "open" and o.tag == "t"
    with pytest.raises(BrokerError, match="US equities only"):
        broker.submit(OrderRequest("XLON_VOD", "buy", 1))
    with pytest.raises(BrokerError, match="unknown instrument"):
        broker.submit(OrderRequest("XNYS_JPM", "buy", 1))
    assert broker.cancel("o-2").status.value == "cancelled" and broker._client.cancelled == ["o-2"]
    with pytest.raises(BrokerError, match="USD"):
        AlpacaBroker("k", "s", base_currency="GBP")


# ---------------------------------------------------------------- runner end to end
def _seed_cache(settings, catalog, ids, start=date(2023, 6, 1), end=date(2024, 6, 28)):
    store = ParquetStore(settings.paths.parquet_dir)
    for k, iid in enumerate(ids):
        store.write_bars(make_bars(catalog[iid], start, end, seed=k + 11, vendor="yfinance"))


def test_paper_runner_end_to_end(tmp_settings, catalog, tmp_path):
    ids = ["XNAS_AAPL", "XNAS_MSFT"]
    _seed_cache(tmp_settings, catalog, ids)
    svc = DataService(tmp_settings, catalog=catalog)
    ledger = Ledger(svc.db)
    state = ledger.create_account(
        "demo", broker="sim", base_currency="USD", initial_cash=1_000_000.0, as_of=date(2024, 6, 18)
    )
    broker = SimBroker(state, {i: catalog[i] for i in ids}, cost_model=CostModel(), ledger=ledger)
    strat = FixedWeights(ids, {"XNAS_AAPL": 0.05, "XNAS_MSFT": 0.04}, rebalance="never")
    runner = PaperRunner(svc, broker, strat, PreTradeRules(), refresh=False)
    r1 = runner.run_session(date(2024, 6, 21))  # 6/20 and 6/21 are replayed (6/19 is a holiday)
    assert r1.processed_sessions == [date(2024, 6, 20), date(2024, 6, 21)] and not r1.fills
    assert len(r1.submitted) == 2 and not r1.blocked and not r1.broker_rejected
    assert all(o.eligible_from == date(2024, 6, 24) for o in r1.submitted)
    assert r1.reconciliation is not None and r1.reconciliation.ok and "submitted" in r1.summary()
    r2 = runner.run_session(date(2024, 6, 24))
    assert len(r2.fills) == 2 and r2.account.n_positions == 2 and r2.account.n_open_orders == 0
    assert not r2.submitted  # 'never' rebalances once
    w = {p.instrument_id: (p.market_value_base or 0) / r2.account.equity for p in broker.positions()}
    assert w["XNAS_AAPL"] == pytest.approx(0.05, abs=0.005) and w["XNAS_MSFT"] == pytest.approx(
        0.04, abs=0.005
    )
    reloaded = ledger.load("demo")
    assert reloaded.as_of == date(2024, 6, 24) and len(reloaded.fills) == 2 and len(reloaded.equity) == 3
    assert reconcile(reloaded, broker).ok
    with pytest.raises(BrokerError, match="before"):
        runner.run_session(date(2024, 6, 20))
    html = build_blotter(reloaded, broker, reconciliation=r2.reconciliation, path=tmp_path / "blotter.html")
    assert "Caveats" in html and "PAPER TRADING" in html and "Positions" in html and "demo" in html

    # a tighter rule set blocks the same orders before they reach the broker
    state2 = ledger.create_account(
        "tight", broker="sim", base_currency="USD", initial_cash=1_000_000.0, as_of=date(2024, 6, 18)
    )
    broker2 = SimBroker(state2, {i: catalog[i] for i in ids}, cost_model=CostModel(), ledger=ledger)
    runner2 = PaperRunner(
        svc,
        broker2,
        FixedWeights(ids, {"XNAS_AAPL": 0.05, "XNAS_MSFT": 0.04}, rebalance="never"),
        PreTradeRules(max_position_pct=1.0),
        refresh=False,
    )
    r = runner2.run_session(date(2024, 6, 21))
    assert len(r.blocked) == 2 and not r.submitted
    assert "max_position_pct" in r.blocked_table()["failures"].iloc[0]
    assert any(e["type"] == "pretrade_blocked" for e in state2.events)

    # manual order through the gate
    report, order, _ = submit_manual(
        runner, OrderRequest("XNAS_AAPL", "buy", 10, tag="manual"), session_date=date(2024, 6, 25)
    )
    assert report.passed and order is not None and order.status.value == "open"
    blocked, none, _ = submit_manual(
        runner, OrderRequest("XNAS_AAPL", "buy", 10_000), session_date=date(2024, 6, 25)
    )
    assert not blocked.passed and none is None
