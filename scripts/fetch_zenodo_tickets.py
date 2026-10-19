"""Download the real-world ticket set used in the paper revision (Oct 2026).

Dataset: L. S. B. Pereira, "Classification of IT Support Tickets", Zenodo,
2022, doi:10.5281/zenodo.7648117 (latest version of the concept record;
v1 is 10.5281/zenodo.7384758). Licence CC BY 4.0. 2,229 real tickets from an
IT support company in the Florianopolis (Brazil) region, typed by the users
who opened them and labelled into 7 categories by three IT professionals.

Only the data files are fetched. The record also ships a Keras model and two
pickles; they are NOT downloaded (unpickling executes code, and nothing here
needs them). code.py is fetched as TEXT, to document the column layout, and
is never imported or executed.

Every file is pinned by the md5 the Zenodo API published for it, and a
mismatch is FATAL -- a silently changed upstream file would otherwise become a
silently changed result. Files land in data/external_zenodo_7648117/raw/,
which holds nothing else and is never mixed with our own artifacts.

    .\\venv\\Scripts\\python.exe scripts/fetch_zenodo_tickets.py
"""

from __future__ import annotations

import argparse
import hashlib
import os
import urllib.request

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "external_zenodo_7648117", "raw")

RECORD = "7648117"
URL = "https://zenodo.org/api/records/{rec}/files/{name}/content"
# md5 as published by https://zenodo.org/api/records/7648117 (2026-10-19).
FILES = {
    "X_train.csv": "0a79f0c2f95fcdb407f24b2c0ad7f7d1",
    "X_test.csv": "02328c4e09a78a41acb22ce17e6e0312",
    "y_train.csv": "90876921486a26a67bc1976da87305d9",
    "y_test.csv": "2f054c0e3d8cbbbd76adf30d13a7710a",
    "distributions.txt": "f5f206a54b2b2a58c820c03349ba8cd1",
    "metrics.txt": "185f22256544b811597e697b6a63f79d",
    "code.py": "811cee209e221033db06587d7d331dce",
}


def die(msg):
    print(f"\n[FATAL] {msg}")
    raise SystemExit(1)


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true",
                    help="re-download files that already exist")
    args = ap.parse_args()
    os.makedirs(RAW_DIR, exist_ok=True)

    for name, expected in FILES.items():
        path = os.path.join(RAW_DIR, name)
        if os.path.exists(path) and not args.force:
            print(f"[cached] {name}")
        else:
            url = URL.format(rec=RECORD, name=name)
            try:
                with urllib.request.urlopen(url, timeout=60) as resp:
                    data = resp.read()
            except Exception as exc:
                die(f"download failed for {name}: {exc}")
            with open(path, "wb") as fh:
                fh.write(data)
            print(f"[fetch]  {name} ({len(data)} bytes)")
        got = md5(path)
        if got != expected:
            die(f"{name}: md5 {got} != pinned {expected}. The upstream file "
                "changed, or the download is corrupt. Do not proceed.")
    print(f"[ok] all {len(FILES)} files match their pinned md5 -> "
          f"{os.path.relpath(RAW_DIR, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
