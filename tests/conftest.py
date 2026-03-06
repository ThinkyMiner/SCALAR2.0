import os
import shutil
import pytest

os.environ.setdefault("SCALAR_API_KEYS", "test-key-abc")
os.environ.setdefault("SCALAR_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
os.environ.setdefault("SCALAR_DATA_DIR", "/tmp/scalar_test_data")


@pytest.fixture(scope="session", autouse=True)
def clean_test_data_dir():
    """Wipe and recreate the test data directory at the start of every test session."""
    data_dir = os.environ["SCALAR_DATA_DIR"]
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir, exist_ok=True)
    yield
