# conftest.py
import asyncio
import pytest


@pytest.fixture(scope="session")
def event_loop():
    """
    Create an event loop per test session (per xdist worker),
    so pytest-xdist doesn't conflict with pytest-asyncio's default loop handling.
    """
    try:
        # Use uvloop if available for better performance
        import uvloop
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    except ImportError:
        pass

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop

    # Gracefully shut down async generators (Python 3.6+)
    loop.run_until_complete(loop.shutdown_asyncgens())

    # On Python 3.9+, also shutdown default executor
    if hasattr(loop, "shutdown_default_executor"):
        loop.run_until_complete(loop.shutdown_default_executor())

    loop.close()
