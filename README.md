# force-vla-basics — Stage 0

Trustworthy force/torque readings from a simulated peg-in-hole task (MuJoCo + robosuite),
and an understanding of what contaminates them. See `PLAN.md` for the full plan and
`docs/FINDINGS.md` for results.

## Quickstart

```bash
make build      # build the Docker image (CPU, headless)
make test       # pytest inside the container
make m0         # API probe -> outputs/m0/probe.json
make m1         # Track A gantry insertion -> outputs/m1/
```

Everything runs inside Docker. `make shell` drops you into the container,
`make run S=scripts/xx.py ARGS="..."` runs one script.
