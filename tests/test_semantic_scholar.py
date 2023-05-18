from io import StringIO

import pytest

from src.semantic_scholar_batch_fetch import (
    chunk_iterator,
    fetch_semantic_scholar_by_ids,
)


@pytest.fixture
def temp_file():
    # Create an in-memory file-like object
    file = StringIO()
    yield file  # Provide the file-like object as the fixture value

    # Optional: Clean up the file-like object after the test
    file.close()


def test_fetch_ids(temp_file):
    ids = [
        "c41eb895616e453dcba1a70c9b942c5063cc656c",
        "b022f2a277a4bf5f42382e86e4380b96340b9e86",
        "f61237db63fb1616fe2c9ff8a81d863a72500a37",
        "59a922212153d3407e658109f36c11a34ee7d283",
        "202cbbf671743aefd380d2f23987bd46b9caaf97",
        "4f847b4ddc105d73bc78f3e7220e6c1f71a7dfb6",
        "1b9c6022598085dd892f360122c0fa4c630b3f18",
        "9ba0186ed40656329c421f55ada7313293e13f17",
        "0f50b7483f1b200ebf88c4dd7698de986399a0f3",
        "0d0eeb46fc5ec778a62bb94aa2ef261b08e6f8c6",
        "09193e19b59fc8f05bee9d6efbfb1607ca5b6501",
    ]

    items = {k: k for k in ids}

    # Fetch the papers from Semantic Scholar via a file.
    results = fetch_semantic_scholar_by_ids(items, "backward", temp_file)
    assert len(results) == 0

    # Fetch the papers from Semantic Scholar.
    values = temp_file.getvalue()
    assert len(ids) == len(values.strip().split("\n"))

    # In memory instead.
    results = fetch_semantic_scholar_by_ids(items, "backward")
    assert len(results) == len(ids)
    assert all([id == r["id"] for id, r in zip(ids, results)])


def test_iterator():
    ids = {"a": "b", "c": "d", "e": "f"}
    for el in chunk_iterator(list(ids.items()), 2):
        keys, items = zip(*el)
        assert keys[0] in ["a", "e"]
