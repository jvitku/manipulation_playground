#!/usr/bin/env python
"""Stage 1 G3: export an expert dataset directory to a robomimic-style HDF5 file.

Example: python scripts/15_export_hdf5.py --data data/expert_arm --out data/hdf5/expert_arm.hdf5
Layout: see fvb.policy.export.
"""

from __future__ import annotations

import argparse
import json

from fvb.policy.export import export_hdf5


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", nargs="+", required=True, help="one or more expert dataset dirs")
    ap.add_argument("--out", nargs="+", required=True, help="one .hdf5 path per --data")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    assert len(args.data) == len(args.out), "--data and --out must pair up"
    for d, o in zip(args.data, args.out, strict=True):
        print(json.dumps(export_hdf5(d, o, args.val_frac, args.seed)))


if __name__ == "__main__":
    main()
