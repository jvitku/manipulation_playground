"""Task registry and task specs.

Importing this package registers all tasks. Later tasks add their modules
below (e.g. `from lab.tasks import t1_xxx  # noqa: F401`) to trigger `register()`.
"""

from lab.tasks import (
    task1_bc_mse,  # noqa: F401
    task2_flow,  # noqa: F401
    task3_flow_bc,  # noqa: F401
    task4_force_token,  # noqa: F401
    task5_tavla,  # noqa: F401
    task6_td3,  # noqa: F401
    task7_td3bc,  # noqa: F401
    task8_residual,  # noqa: F401
    task_attention,  # noqa: F401
    task_embed,  # noqa: F401
)
