from __future__ import annotations

import pandas as pd
import pytest

from textclf.config import DataConfig
from textclf.data import make_splits

TEMPLATES = {
    "card_arrival": ["where is my new card", "my card has not arrived yet", "still waiting for the card to come"],
    "exchange_rate": ["what exchange rate do you use", "how is the currency rate set", "exchange rate for euros today"],
    "lost_card": ["i lost my card", "my card is missing please block it", "card was stolen yesterday"],
}


def _frame(n_per: int, offset: int) -> pd.DataFrame:
    rows = []
    for intent, ts in TEMPLATES.items():
        for i in range(n_per):
            rows.append({"text": f"{ts[i % 3]} number {i + offset}", "category": intent})
    return pd.DataFrame(rows)


@pytest.fixture
def raw_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    return _frame(20, 0), _frame(6, 1000)


@pytest.fixture
def splits(raw_frames):
    tr, te = (df.rename(columns={"category": "intent"}) for df in raw_frames)
    return make_splits(tr, te, DataConfig(val_fraction=0.2, split_seed=0))
