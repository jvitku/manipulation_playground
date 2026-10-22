"""Entry point of a job subprocess: `python -m lab.run_job {train|eval} <job_dir>`."""

import sys
import traceback
from pathlib import Path

from lab.core.jobs import read_status
from lab.core.runctx import RunContext
from lab.tasks import registry


def main(argv: list[str] | None = None) -> int:
    kind, job_dir = (argv if argv is not None else sys.argv[1:])[:2]
    if kind not in ("train", "eval"):
        raise SystemExit(f"kind must be train|eval, got {kind!r}")
    ctx = RunContext(Path(job_dir))
    ctx.set_status(state="running")
    try:
        spec = registry.get(ctx.task_id)
        (spec.train if kind == "train" else spec.evaluate)(ctx)
    except BaseException as e:  # noqa: BLE001 - any death must land in status.json
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        ctx.fail(f"{type(e).__name__}: {e}\n{tb[-1500:]}")
        return 1
    if read_status(ctx.dir).get("state") not in ("done", "failed"):
        ctx.done()
    return 0


if __name__ == "__main__":
    sys.exit(main())
