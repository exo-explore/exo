"""Tests for exo.api.billing — API key management and quota enforcement."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from exo.api.billing import (
    PLANS,
    check_quota,
    check_rate_limit,
    create_key,
    get_key_info,
    record_usage,
    revoke_key,
    tokens_used_this_month,
)


@pytest.fixture(autouse=True)
def _tmp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]
    """Redirect billing DB to a temp file for each test."""
    import exo.api.billing as billing_module

    monkeypatch.setattr(billing_module, "_DB_PATH", tmp_path / "billing.db")


# ── Plans ─────────────────────────────────────────────────────────────────────

class TestPlans:
    def test_free_plan_exists(self) -> None:
        assert "free" in PLANS

    def test_starter_plan_has_positive_price(self) -> None:
        assert PLANS["starter"]["price_usd"] > 0

    def test_enterprise_unlimited(self) -> None:
        assert PLANS["enterprise"]["monthly_tokens"] == -1
        assert PLANS["enterprise"]["rpm"] == -1

    def test_all_plans_have_required_keys(self) -> None:
        for name, cfg in PLANS.items():
            assert "monthly_tokens" in cfg, f"plan {name} missing monthly_tokens"
            assert "rpm" in cfg, f"plan {name} missing rpm"
            assert "price_usd" in cfg, f"plan {name} missing price_usd"


# ── Key management ────────────────────────────────────────────────────────────

class TestKeyManagement:
    def test_create_key_returns_raw_key(self) -> None:
        key = create_key(client_id="test@example.com", plan="free")
        assert key.startswith("exo-")

    def test_created_key_is_retrievable(self) -> None:
        key = create_key(client_id="alice@example.com", plan="starter")
        info = get_key_info(key)
        assert info is not None
        assert info["client_id"] == "alice@example.com"
        assert info["plan"] == "starter"
        assert info["active"] == "1"

    def test_unknown_key_returns_none(self) -> None:
        assert get_key_info("exo-doesnotexist") is None

    def test_revoke_key(self) -> None:
        key = create_key(client_id="bob@example.com", plan="free")
        assert get_key_info(key) is not None
        ok = revoke_key(key)
        assert ok is True
        # revoked key should not be found
        assert get_key_info(key) is None

    def test_revoke_nonexistent_key(self) -> None:
        ok = revoke_key("exo-doesnotexist")
        assert ok is False

    def test_key_hash_not_exposed(self) -> None:
        key = create_key(client_id="carol@example.com", plan="pro")
        info = get_key_info(key)
        assert info is not None
        # raw key should NOT be stored
        assert info["key_hash"] != key
        # should be a sha256 hex
        expected_hash = hashlib.sha256(key.encode()).hexdigest()
        assert info["key_hash"] == expected_hash

    def test_notes_stored(self) -> None:
        key = create_key(client_id="dave@example.com", plan="free", notes="test contract")
        info = get_key_info(key)
        assert info is not None
        assert info["notes"] == "test contract"


# ── Quota enforcement ─────────────────────────────────────────────────────────

class TestQuotaEnforcement:
    def test_free_plan_within_quota(self) -> None:
        key = create_key(client_id="user@example.com", plan="free")
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        allowed, reason = check_quota(key_hash, "free")
        assert allowed is True
        assert reason == "ok"

    def test_enterprise_always_allowed(self) -> None:
        key = create_key(client_id="bigcorp@example.com", plan="enterprise")
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        allowed, _ = check_quota(key_hash, "enterprise")
        assert allowed is True

    def test_quota_exceeded_blocks(self) -> None:
        key = create_key(client_id="heavyuser@example.com", plan="free")
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        # Record usage that exceeds free plan limit (100K tokens)
        record_usage(
            key_hash=key_hash,
            endpoint="/v1/chat/completions",
            input_tokens=60_000,
            output_tokens=50_000,  # total 110K > 100K limit
        )
        allowed, reason = check_quota(key_hash, "free")
        assert allowed is False
        assert "quota exceeded" in reason.lower()

    def test_tokens_used_this_month_zero_initially(self) -> None:
        key = create_key(client_id="fresh@example.com", plan="free")
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        used = tokens_used_this_month(key_hash)
        assert used == 0

    def test_tokens_used_counts_correctly(self) -> None:
        key = create_key(client_id="counter@example.com", plan="free")
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        record_usage(
            key_hash=key_hash,
            endpoint="/v1/chat/completions",
            input_tokens=1_000,
            output_tokens=500,
        )
        used = tokens_used_this_month(key_hash)
        assert used == 1_500


# ── Rate limiting ─────────────────────────────────────────────────────────────

class TestRateLimiting:
    def test_enterprise_no_rate_limit(self) -> None:
        key_hash = "fake_enterprise_hash"
        for _ in range(200):
            check_rate_limit(key_hash, "enterprise")
        final_allowed, _ = check_rate_limit(key_hash, "enterprise")
        assert final_allowed is True

    def test_free_plan_rate_limited(self) -> None:
        key = create_key(client_id="rapid@example.com", plan="free")
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        rpm_limit = PLANS["free"]["rpm"]  # 10

        for _ in range(rpm_limit):
            check_rate_limit(key_hash, "free")

        # One more should be rate limited
        final_allowed, reason = check_rate_limit(key_hash, "free")
        assert final_allowed is False
        assert "rate limit" in reason.lower()


# ── Usage recording ───────────────────────────────────────────────────────────

class TestUsageRecording:
    def test_record_usage_persists(self) -> None:
        key = create_key(client_id="recorder@example.com", plan="starter")
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        record_usage(
            key_hash=key_hash,
            endpoint="/v1/chat/completions",
            input_tokens=100,
            output_tokens=200,
            model="mlx-community/Qwen3-8B-4bit",
            request_id="req_test_001",
        )
        used = tokens_used_this_month(key_hash)
        assert used == 300

    def test_multiple_records_accumulate(self) -> None:
        key = create_key(client_id="multi@example.com", plan="pro")
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        for _ in range(5):
            record_usage(
                key_hash=key_hash,
                endpoint="/v1/chat/completions",
                input_tokens=100,
                output_tokens=100,
            )
        used = tokens_used_this_month(key_hash)
        assert used == 1_000
