import os
import uuid

import pytest
from exo_rs import SessionHandle


ZENOH_PORT = 52414
DISCOVERY_PORT = 52413


@pytest.fixture(scope="module")
def storage():
    node_id = os.urandom(16).hex().rstrip("0")

    session_handle, _nh = SessionHandle.new(
        node_id,
        ZENOH_PORT,
        DISCOVERY_PORT,
    )

    return session_handle.storage_interface()


@pytest.mark.asyncio
async def test_storage_get_missing_key_returns_none(storage):
    key = f"tests/storage/{uuid.uuid4().hex}/missing"

    value = await storage.get(key)
    assert value is None


@pytest.mark.asyncio
async def test_storage_put_then_get_returns_value(storage):
    key = f"tests/storage/{uuid.uuid4().hex}/value"
    expected = "hello storage"

    await storage.put(key, expected)
    assert await storage.get(key) == expected


@pytest.mark.asyncio
async def test_storage_put_overwrites_value(storage):
    key = f"tests/storage/{uuid.uuid4().hex}/overwrite"

    await storage.put(key, "old")
    await storage.put(key, "new")
    assert await storage.get(key) == "new"


@pytest.mark.asyncio
async def test_storage_put_overwrites_value(storage):
    key = f"tests/storage/{uuid.uuid4().hex}/overwrite"

    await storage.put(key, "old")
    await storage.delete(key)
    assert await storage.get(key) == None


@pytest.mark.asyncio
async def test_storage_get_rejects_wildcard_key(storage):
    with pytest.raises(ValueError, match="only supports fixed keys"):
        await storage.get("tests/storage/*")


@pytest.mark.asyncio
async def test_storage_put_rejects_wildcard_key(storage):
    with pytest.raises(ValueError, match="only supports fixed keys"):
        await storage.put("tests/storage/*", "value")
