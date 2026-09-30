#!/usr/bin/env python
"""Convert recorded SortBoltsNuts demos to a LeRobotDataset (VLA image).

    python scripts/21_to_lerobot.py --raw outputs/sort_demos_pilot \
        --root outputs/lerobot/sort_synth \
        --mode segments
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", required=True, help="directory with ep_*/ episodes")
    ap.add_argument(
        "--root", required=True, help="output LeRobotDataset directory (must not exist)"
    )
    ap.add_argument("--repo-id", default="local/sort_synth")
    ap.add_argument("--mode", choices=("episode", "segments"), default="segments")
    args = ap.parse_args()

    from fvb.vla.lerobot_convert import convert

    eps = sorted(Path(args.raw).glob("ep_*"))
    out = convert(eps, Path(args.root), args.repo_id, args.mode)
    print(json.dumps({"raw_episodes": len(eps), **out}))


if __name__ == "__main__":
    main()
