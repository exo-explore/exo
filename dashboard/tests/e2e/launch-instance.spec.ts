import { test, expect } from "@playwright/test";
import {
  waitForTopologyLoaded,
  waitForModelCards,
  selectModelFromLaunchDropdown,
} from "../helpers/wait-for-ready";

test.describe("Launch Instance", () => {
  test("should launch Qwen3-0.6B-4bit model", async ({ page }) => {
    await page.goto("/");
    await waitForTopologyLoaded(page);

    // First select the model from the dropdown (model cards appear after selection)
    await selectModelFromLaunchDropdown(page, /qwen.*0\.6b/i);

    // Now wait for model cards to appear
    await waitForModelCards(page);

    // Find and click on the model card (should already be filtered to Qwen)
    const modelCard = page.locator('[data-testid="model-card"]').first();
    await expect(modelCard).toBeVisible({ timeout: 10000 });

    // Click the launch button
    const launchButton = modelCard.locator('[data-testid="launch-button"]');
    await launchButton.click();

    // Wait for the model to start (status should change to READY or show download progress)
    // The model may need to download first, so we wait with a longer timeout
    await expect(
      page
        .locator('[data-testid="instance-status"]')
        .filter({ hasText: /READY|downloading/i })
        .first(),
    ).toBeVisible({ timeout: 300000 }); // 5 minutes for download
  });
});
