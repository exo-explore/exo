import { test, expect } from "@playwright/test";
import { waitForTopologyLoaded } from "../helpers/wait-for-ready";

test.describe("Homepage", () => {
  test("should load and display key elements", async ({ page }) => {
    await page.goto("/");
    await waitForTopologyLoaded(page);

    // Verify key UI elements are present
    await expect(page.locator('[data-testid="topology-graph"]').first()).toBeVisible();
    await expect(page.locator('[data-testid="chat-input"]')).toBeVisible();
    await expect(page.locator('[data-testid="send-button"]')).toBeVisible();
  });
});
