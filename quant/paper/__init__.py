"""Paper trading: broker contract, SimBroker, Alpaca paper adapter, pre-trade gate, ledger, runner, blotter.

There is no live trading here. `quant.config.live_gate` is the only live path and it raises.
"""

from quant.paper.alpaca_broker import AlpacaBroker
from quant.paper.blotter import build_blotter
from quant.paper.broker import AccountSnapshot, Broker, BrokerFill, BrokerOrder, OrderRequest, Position
from quant.paper.ledger import AccountState, Ledger, PositionState, ledger_invariants
from quant.paper.pretrade import Check, PreTradeReport, PreTradeRules, check_order
from quant.paper.reconcile import ReconciliationReport, reconcile
from quant.paper.runner import PaperRunner, SessionReport, submit_manual
from quant.paper.sim_broker import SimBroker

__all__ = [
    "AccountSnapshot",
    "AccountState",
    "AlpacaBroker",
    "Broker",
    "BrokerFill",
    "BrokerOrder",
    "Check",
    "Ledger",
    "OrderRequest",
    "PaperRunner",
    "Position",
    "PositionState",
    "PreTradeReport",
    "PreTradeRules",
    "ReconciliationReport",
    "SessionReport",
    "SimBroker",
    "build_blotter",
    "check_order",
    "ledger_invariants",
    "reconcile",
    "submit_manual",
]
