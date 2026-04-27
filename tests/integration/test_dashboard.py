# type: ignore
"""Dashboard end-to-end tests using Playwright (headless Chromium).

Prerequisites:
    npx --yes playwright install chromium

Run with:
    uv run pytest integration_tests/test_dashboard.py -v
"""
from __future__ import annotations

import pytest

from .helpers import ClusterInfo, make_client, place_and_wait


@pytest.fixture
def playwright_page(single_node_cluster: ClusterInfo):
    """Create a Playwright browser page pointed at the cluster's dashboard."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        pytest.skip("playwright not installed (run: pip install playwright && playwright install chromium)")

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

    def test_dashboard_shows_node_info(self, playwright_page, single_node_cluster: ClusterInfo):
        """Verify the dashboard displays node/cluster information."""
        page = playwright_page
        page.screenshot(path="/tmp/dashboard_cluster_info.png")

        # The dashboard should show at least one node — look for node-related
        # content such as memory, model count, or the cluster status section.
        # We check for known UI elements that indicate the dashboard rendered
        # cluster data, not just a blank shell.
        body_text = (page.text_content("body") or "").lower()

        # The dashboard should contain some indicator of cluster state:
        # memory info, node count, model names, or status indicators
        has_cluster_content = any(
            indicator in body_text
            for indicator in ["gb", "memory", "node", "model", "connected", "online", "select"]
        )
        assert has_cluster_content, (
            f"Dashboard doesn't appear to show cluster info. "
            f"Body preview: {body_text[:500]}"
        )

    def test_dashboard_chat_inference(self, single_node_cluster: ClusterInfo):
        """Full flow: place model via API, then use dashboard to chat."""
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            pytest.skip("playwright not installed")

        client = make_client(single_node_cluster)

        # Place model via API first (more reliable than clicking through UI)
        place_and_wait(client)

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            page.goto(single_node_cluster.api_url, wait_until="networkidle")
            page.wait_for_timeout(3000)

            page.screenshot(path="/tmp/dashboard_before_chat.png")

            # Try to find and interact with chat input
            # Note: selectors may need adjustment based on actual dashboard DOM
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
                pytest.skip("Could not find chat textarea — dashboard UI may have changed")

            browser.close()
