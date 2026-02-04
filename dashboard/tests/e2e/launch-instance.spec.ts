/// <reference types="node" />
import { test, expect } from "@playwright/test";
import { waitForTopologyLoaded } from "../helpers/wait-for-ready";

test.describe("Launch Model Instance", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
    await waitForTopologyLoaded(page);
  });

  test("should display topology graph on page load", async ({ page }) => {
    const topologyGraph = page.locator('[data-testid="topology-graph"]');
    await expect(topologyGraph).toBeVisible();
  });

  test("should display model dropdown on welcome page", async ({ page }) => {
    const modelDropdown = page.locator('[data-testid="model-dropdown"]');
    await expect(modelDropdown).toBeVisible({ timeout: 15000 });
  });

  test("should display model cards when model is selected", async ({
    page,
  }) => {
    const modelDropdown = page.locator('[data-testid="model-dropdown"]');
    await expect(modelDropdown).toBeVisible({ timeout: 15000 });

    // Wait for API to respond with placement previews
    await page.waitForTimeout(5000);

    // Check if launch button is visible (indicates model card is showing)
    const launchButton = page.locator('[data-testid="launch-button"]').first();
    let hasLaunchButton = await launchButton
      .isVisible({ timeout: 5000 })
      .catch(() => false);

    if (!hasLaunchButton) {
      // No model selected yet - click dropdown and select one
      await modelDropdown.click();
      await page.waitForTimeout(1000);

      const modelOption = page
        .locator('[data-testid="model-option"]:not([disabled])')
        .first();
      await expect(modelOption).toBeVisible({ timeout: 5000 });

      await modelOption.click();
      await page.waitForTimeout(3000);
    }

    await expect(launchButton).toBeVisible({ timeout: 30000 });
  });

  test("should launch a model instance", async ({ page }) => {
    // Skip in CI - requires downloading a model which is too slow
    test.skip(!!process.env.CI, "Skipped in CI - requires model download");
    test.setTimeout(300000); // 5 minutes

    const modelDropdown = page.locator('[data-testid="model-dropdown"]');
    await expect(modelDropdown).toBeVisible({ timeout: 15000 });

    // Wait for placement previews to load
    await page.waitForTimeout(3000);

    // Check if launch button is visible
    const launchButton = page
      .locator('[data-testid="launch-button"]:not([disabled])')
      .first();
    let hasLaunchButton = await launchButton
      .isVisible({ timeout: 10000 })
      .catch(() => false);

    if (!hasLaunchButton) {
      // Select a model first - prefer Qwen3-0.6B (smallest)
      await modelDropdown.click();
      await page.waitForTimeout(500);

      // Try to find Qwen3-0.6B-4bit specifically (smallest model), otherwise take first available
      const qwenOption = page
        .locator('[data-testid="model-option"]', {
          hasText: /qwen.*0\.6b.*4bit/i,
        })
        .first();
      const hasQwen = await qwenOption
        .isVisible({ timeout: 2000 })
        .catch(() => false);

      if (hasQwen) {
        await qwenOption.click();
      } else {
        const modelOption = page
          .locator('[data-testid="model-option"]:not([disabled])')
          .first();
        await expect(modelOption).toBeVisible({ timeout: 5000 });
        await modelOption.click();
      }

      await expect(launchButton).toBeVisible({ timeout: 30000 });
    }

    // Click launch
    await launchButton.click();

    // Wait for instance to start - could be downloading, loading, or ready
    await expect(async () => {
      const isDownloading = await page
        .locator("text=DOWNLOADING")
        .first()
        .isVisible();
      const isReady = await page.locator("text=READY").first().isVisible();
      const isLoading = await page.locator("text=LOADING").first().isVisible();
      const hasLaunching = await launchButton
        .textContent()
        .then((t) => t?.includes("LAUNCHING"));
      expect(
        isDownloading || isReady || isLoading || hasLaunching,
      ).toBeTruthy();
    }).toPass({ timeout: 30000 });

    // Wait for instance to be fully ready (this may take a while if downloading)
    await expect(page.locator("text=READY").first()).toBeVisible({
      timeout: 240000,
    }); // 4 min for download
  });
});
