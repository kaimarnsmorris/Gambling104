"""Where the 5 m harness keeps its panel. Kept in one place so a harness move is a
one-line change here rather than a search across the export and the scorecard."""
from pathlib import Path

BACKTESTING = Path(__file__).resolve().parents[2] / "backtesting_5m"
PANEL = BACKTESTING / "data" / "book_5m_100ms.parquet"
STRIKES = BACKTESTING / "data" / "strikes_5m.parquet"
FAIR_DIR = BACKTESTING / "data" / "fair"
