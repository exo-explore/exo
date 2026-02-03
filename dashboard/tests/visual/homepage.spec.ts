import { test, expect } from "@playwright/test";
import { waitForTopologyLoaded } from "../helpers/wait-for-ready";

test.describe("Homepage Visual Snapshots", () => {
  test("homepage with topology", async ({ page }) => {
    await page.goto("/");
    await waitForTopologyLoaded(page);

    // Wait for the page to fully render
    await page.waitForTimeout(1000);

    await expect(page).toHaveScreenshot("homepage.png", {
      fullPage: true,
    });
  });

  test("topology graph", async ({ page }) => {
    await page.goto("/");
    await waitForTopologyLoaded(page);

    await page.waitForTimeout(1000);

    const topologyGraph = page
      .locator('[data-testid="topology-graph"]')
      .first();
    await expect(topologyGraph).toBeVisible();

    await expect(topologyGraph).toHaveScreenshot("topology-graph.png");
  });
});
