"""Cumulative net PnL over the ordered market sequence."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def cumulative_pnl(markets, path):
    m = markets.sort_values("open_ts")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(range(len(m)), m["pnl_net"].fillna(0.0).cumsum(), lw=1.2)
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.set_xlabel("market (chronological)")
    ax.set_ylabel("cumulative net PnL, USD")
    ax.set_title("Cumulative net PnL")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
