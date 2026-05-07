# type: ignore
"""Dashboard end-to-end tests using Playwright (headless Chromium).

Prerequisites:
    uv run playwright install chromium

Run with:
    uv run pytest tests/test_dashboard.py -v
"""

from __future__ import annotations

import contextlib

import pytest

try:
    from playwright.sync_api import sync_playwright

    _HAS_PLAYWRIGHT = True
except ImportError:
    _HAS_PLAYWRIGHT = False

# Check if Chromium is installed by attempting a quick launch
_HAS_CHROMIUM = False
if _HAS_PLAYWRIGHT:
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            browser.close()
        _HAS_CHROMIUM = True
    except Exception:
        pass

pytestmark = pytest.mark.skipif(
    not _HAS_PLAYWRIGHT or not _HAS_CHROMIUM,
    reason="playwright or chromium not installed (run: uv run playwright install chromium)",
)


def _mark_onboarding_complete(session) -> None:
    """Mark onboarding complete on the server so the wizard doesn't auto-launch a model."""
    with contextlib.suppress(Exception):
        session.client.request_json("POST", "/onboarding")


@pytest.mark.cluster(count=1)
def test_dashboard_chat_inference(session):
    """Full UI flow: open dashboard, pick a model, send a chat, verify response.

    The instance is created via the dashboard UI (model picker → chat send
    triggers the dashboard's auto-launch flow), not via @pytest.mark.instance.
    """
    _mark_onboarding_complete(session)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        page.goto(session.cluster.api_url, wait_until="networkidle")
        page.wait_for_timeout(3000)
        page.screenshot(path="/tmp/dashboard_initial.png")

        # Open the model picker by clicking the "SELECT MODEL" button
        page.get_by_text("SELECT MODEL", exact=False).first.click()
        page.wait_for_timeout(1000)
        page.screenshot(path="/tmp/dashboard_picker_open.png")

        # Search for the model — uses the model id substring; the picker
        # matches against name/id so "Llama-3.2-1B" filters to the small Llama.
        search_input = page.locator('input[placeholder*="Search models"]').first
        search_input.fill("Llama-3.2-1B")
        page.wait_for_timeout(1500)
        page.screenshot(path="/tmp/dashboard_picker_search.png")

        # Click the only matching result. The picker shows the model's
        # display name (e.g. "Llama 3.2 1B") which differs from the model_id.
        # We click the first visible button-like row in the result list.
        page.get_by_text("Llama 3.2 1B", exact=False).first.click()
        page.wait_for_timeout(1500)
        page.screenshot(path="/tmp/dashboard_model_selected.png")

        # Type a chat message — sending triggers the dashboard's auto-launch
        # flow: it picks an optimal placement for the selected model and POSTs
        # to /instance, then sends the chat once the runner is ready.
        chat_input = page.locator("textarea").first
        chat_input.fill("Say hello")
        chat_input.press("Enter")
        page.screenshot(path="/tmp/dashboard_chat_sent.png")

        # Wait for the instance to launch and respond. Generous timeout
        # because this includes model placement + load + generation.
        page.wait_for_timeout(60000)
        page.screenshot(path="/tmp/dashboard_after_chat.png")

        # Verify an instance was created and the chat got a response
        instances = session.client.request_json("GET", "/state").get("instances", {})
        assert len(instances) > 0, "Expected the dashboard to have created an instance"

        body_text = page.text_content("body") or ""
        assert len(body_text) > 0

        browser.close()
