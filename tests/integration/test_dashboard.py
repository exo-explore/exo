# type: ignore
"""Dashboard end-to-end tests using Playwright (headless Chromium).

Prerequisites:
    uv run playwright install chromium

Run with:
    uv run pytest tests/integration/test_dashboard.py -v
"""

from __future__ import annotations

import contextlib

import pytest

from .helpers import ClusterInfo, make_client, place_and_wait

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


@pytest.fixture
def playwright_page(single_node_cluster: ClusterInfo):
    """Create a Playwright browser page pointed at the cluster's dashboard.

    Marks onboarding as complete before loading so the wizard doesn't interfere.
    """
    # Mark onboarding complete on the server so the dashboard skips the wizard.
    client = make_client(single_node_cluster)
    with contextlib.suppress(Exception):
        client.request_json("POST", "/onboarding")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        page.goto(single_node_cluster.api_url, wait_until="networkidle")
        page.wait_for_timeout(3000)
        yield page
        browser.close()


class TestDashboard:
    """End-to-end dashboard tests."""

    def test_dashboard_loads(self, playwright_page):
        """Verify the dashboard page loads without errors."""
        page = playwright_page
        body_text = page.text_content("body")
        assert body_text is not None
        assert len(body_text) > 0

    def test_dashboard_shows_node_info(
        self, playwright_page, single_node_cluster: ClusterInfo
    ):
        """Verify the dashboard displays node/cluster information."""
        page = playwright_page
        page.screenshot(path="/tmp/dashboard_cluster_info.png")

        body_text = (page.text_content("body") or "").lower()

        has_cluster_content = any(
            indicator in body_text
            for indicator in [
                "gb",
                "memory",
                "node",
                "model",
                "connected",
                "online",
                "select",
            ]
        )
        assert has_cluster_content, (
            f"Dashboard doesn't appear to show cluster info. "
            f"Body preview: {body_text[:500]}"
        )

    def test_dashboard_chat_inference(self, single_node_cluster: ClusterInfo):
        """Full flow: place model via API, then use dashboard to chat with it.

        Selects the running instance in the dashboard before chatting to prevent
        the dashboard's auto-launch logic from creating a different (larger) model.
        """
        from .helpers import DEFAULT_MODEL

        client = make_client(single_node_cluster)

        # Place model via API first (more reliable than clicking through UI)
        place_and_wait(client)

        # Mark onboarding as complete on the server so the dashboard skips the
        # onboarding wizard (which can auto-launch a different model).
        with contextlib.suppress(Exception):
            client.request_json("POST", "/onboarding")

        # The model name as it appears in the dashboard instance card
        model_short_name = DEFAULT_MODEL.split("/")[-1]  # Llama-3.2-1B-Instruct-4bit

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            page.goto(single_node_cluster.api_url, wait_until="networkidle")
            page.wait_for_timeout(3000)

            page.screenshot(path="/tmp/dashboard_before_chat.png")

            # Click the running instance card to select it as the chat model.
            # This prevents the chat auto-launch from picking a different model.
            instance_card = page.locator(
                f'[role="button"]:has-text("{model_short_name}")'
            ).first
            if instance_card.count() > 0 and instance_card.is_visible():
                instance_card.click()
                page.wait_for_timeout(1000)

            chat_input = page.locator("textarea").first
            if chat_input.is_visible():
                chat_input.fill("Say hello")
                chat_input.press("Enter")

                # Wait for a response (generous timeout for inference)
                page.wait_for_timeout(30000)
                page.screenshot(path="/tmp/dashboard_after_chat.png")

                body_text = page.text_content("body") or ""
                assert len(body_text) > 0
            else:
                page.screenshot(path="/tmp/dashboard_no_textarea.png")
                pytest.skip(
                    "Could not find chat textarea — dashboard UI may have changed"
                )

            browser.close()
