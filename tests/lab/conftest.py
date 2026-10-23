import pytest

from lab.tasks import registry


@pytest.fixture
def isolated_registry():
    """Restore the task registry after a test that registers throw-away specs."""
    saved = dict(registry._TASKS)
    yield
    registry._TASKS.clear()
    registry._TASKS.update(saved)
