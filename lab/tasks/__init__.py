"""Task registry and task specs.

Importing this package registers all tasks. Later tasks add their modules
below (e.g. `from lab.tasks import t1_xxx  # noqa: F401`) to trigger `register()`.
"""

from lab.tasks import (
    task1_bc_mse,  # noqa: F401
    task2_flow,  # noqa: F401
)
