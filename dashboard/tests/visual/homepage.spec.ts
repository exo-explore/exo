import { test, expect } from "@playwright/test";
import { createRequire } from "module";
const require = createRequire(import.meta.url);
const mockStates = require("../fixtures/mock-state.json");

test.describe("Homepage Visual Snapshots", () => {
  test("empty state - no nodes", async ({ page }) => {
    // Mock the state API to return empty state
    await page.route("**/state", (route) => {
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(mockStates.emptyState),
      });
    });

    await page.goto("/");
    await page.waitForLoadState("networkidle");

    // Wait for the page to stabilize
    await page.waitForTimeout(500);

    await expect(page).toHaveScreenshot("homepage-empty-state.png", {
      fullPage: true,
    });
  });

  test("single node connected", async ({ page }) => {
    // Mock the state API to return single node state
    await page.route("**/state", (route) => {
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(mockStates.singleNodeState),
      });
    });

    await page.goto("/");
    await page.waitForLoadState("networkidle");

    // Wait for the topology to render
    await page.waitForTimeout(1000);

    await expect(page).toHaveScreenshot("homepage-single-node.png", {
      fullPage: true,
    });
  });

  test("multiple nodes in topology", async ({ page }) => {
    // Mock the state API to return multi-node state
    await page.route("**/state", (route) => {
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(mockStates.multiNodeState),
      });
    });

    await page.goto("/");
    await page.waitForLoadState("networkidle");

    // Wait for the topology to render
    await page.waitForTimeout(1000);

    await expect(page).toHaveScreenshot("homepage-multi-node.png", {
      fullPage: true,
    });
  });

  test("topology graph element", async ({ page }) => {
    await page.route("**/state", (route) => {
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(mockStates.singleNodeState),
      });
    });

    await page.goto("/");
    await page.waitForLoadState("networkidle");
    await page.waitForTimeout(1000);

    const topologyGraph = page
      .locator('[data-testid="topology-graph"]')
      .first();
    await expect(topologyGraph).toBeVisible();

    await expect(topologyGraph).toHaveScreenshot(
      "topology-graph-single-node.png",
    );
  });
});
