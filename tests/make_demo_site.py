"""Build a throw-away copy of the site filled with SYNTHETIC data, for previewing the page offline.

    python tests/make_demo_site.py /path/to/out_dir

Never point this at docs/: the data is random and must not be published as real.
"""
import os
import shutil
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))

import build_site  # noqa: E402
from synth import make_fundamentals, make_prices  # noqa: E402


def main(out: str) -> None:
    os.makedirs(os.path.join(out, "data"), exist_ok=True)
    shutil.copy(os.path.join(HERE, "..", "docs", "index.html"), os.path.join(out, "index.html"))
    data_dir = os.path.join(out, "data")
    spy, qqq = make_prices(seed=2, drift=0.0005, vol=0.009), make_prices(seed=3, drift=0.0007, vol=0.012)
    specs = [("DEMOUP", 11, 0.0012, True), ("DEMODN", 12, -0.0008, False), ("DEMOCHOP", 13, 0.0001, True)]
    rows = []
    for sym, seed, drift, strong in specs:
        f = make_fundamentals(strong) | {"symbol": sym, "name": f"{sym.title()} (synthetic)"}
        p = build_site.build_payload(sym, make_prices(seed=seed, drift=drift), spy, qqq, f)
        build_site.write_ticker(p, data_dir)
        rows.append(build_site.summary_row(p))
    build_site.update_index(rows, data_dir)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1])
