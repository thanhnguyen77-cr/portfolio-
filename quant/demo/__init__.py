"""Offline end-to-end demo on a SYNTHETIC mini-dataset (`quant demo`)."""

from quant.demo.dataset import (
    DEMO_END,
    DEMO_SEED,
    DEMO_START,
    DEMO_VENDOR,
    SYNTHETIC_NOTICE,
    DemoAdapter,
    DemoManifest,
    build_demo_dataset,
    demo_instruments,
    generate_bars,
    generate_macro,
)
from quant.demo.pipeline import Demo, DemoResult, StepResult

__all__ = [
    "DEMO_END",
    "DEMO_SEED",
    "DEMO_START",
    "DEMO_VENDOR",
    "SYNTHETIC_NOTICE",
    "Demo",
    "DemoAdapter",
    "DemoManifest",
    "DemoResult",
    "StepResult",
    "build_demo_dataset",
    "demo_instruments",
    "generate_bars",
    "generate_macro",
]
