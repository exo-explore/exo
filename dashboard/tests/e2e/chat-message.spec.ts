import { test, expect } from "@playwright/test";
import {
  waitForTopologyLoaded,
  waitForModelCards,
  waitForChatReady,
  waitForAssistantMessage,
  sendChatMessage,
} from "../helpers/wait-for-ready";

test.describe("Chat Message", () => {
  test("should send a message and receive a response", async ({ page }) => {
    // Increase timeout for this test since it involves model loading and inference
    test.setTimeout(600000); // 10 minutes

    await page.goto("/");
    await waitForTopologyLoaded(page);
    await waitForModelCards(page);

    // Find and click on Qwen3-0.6B-4bit model card to launch it
    const modelCard = page
      .locator('[data-testid="model-card"]')
      .filter({ hasText: /qwen.*0\.6b.*4bit/i })
      .first();
    await expect(modelCard).toBeVisible({ timeout: 10000 });

    // Click the launch button
    const launchButton = modelCard.locator('[data-testid="launch-button"]');
    await launchButton.click();

    // Wait for the model to be ready (may take time to download)
    await expect(
      page
        .locator('[data-testid="instance-status"]')
        .filter({ hasText: /READY/i })
        .first(),
    ).toBeVisible({ timeout: 300000 }); // 5 minutes for download

    // Wait for chat to be ready
    await waitForChatReady(page);

    // Select the model in the chat selector if needed
    const modelSelector = page.locator('[data-testid="chat-model-selector"]');
    if (await modelSelector.isVisible()) {
      await modelSelector.click();
      await page.locator("text=/qwen.*0\\.6b/i").first().click();
    }

    // Send a simple message
    await sendChatMessage(page, "What is 2+2?");

    // Wait for assistant response
    await waitForAssistantMessage(page, 120000); // 2 minutes for inference

    // Verify the assistant message is visible
    const assistantMessage = page
      .locator('[data-testid="assistant-message"]')
      .last();
    await expect(assistantMessage).toBeVisible();

    // The response should contain something (not empty)
    const messageContent = await assistantMessage.textContent();
    expect(messageContent).toBeTruthy();
    expect(messageContent!.length).toBeGreaterThan(0);
  });
});
