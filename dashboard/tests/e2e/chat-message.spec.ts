import { test, expect } from "@playwright/test";
import { waitForTopologyLoaded } from "../helpers/wait-for-ready";

test.describe("Chat Message Flow", () => {
  // This test suite requires a running instance
  // It will launch one if needed

  test("should send a chat message and receive a response", async ({
    page,
  }) => {
    // This test may need to launch an instance, so give it plenty of time
    test.setTimeout(600000); // 10 minutes

    await page.goto("/");
    await waitForTopologyLoaded(page);

    // Check if we already have a running instance with chat available
    const chatInput = page.locator('[data-testid="chat-input"]');
    let hasChatReady = await chatInput
      .isVisible({ timeout: 5000 })
      .catch(() => false);

    // Check if there's a ready instance (use first() in case multiple exist)
    const readyIndicator = page.locator("text=READY").first();
    let hasReady = await readyIndicator
      .isVisible({ timeout: 2000 })
      .catch(() => false);

    if (!hasReady) {
      // Need to launch an instance first
      const modelDropdown = page.locator('[data-testid="model-dropdown"]');
      await expect(modelDropdown).toBeVisible({ timeout: 15000 });

      // Wait for placement previews
      await page.waitForTimeout(3000);

      const launchButton = page
        .locator('[data-testid="launch-button"]:not([disabled])')
        .first();
      let hasLaunchButton = await launchButton
        .isVisible({ timeout: 10000 })
        .catch(() => false);

      if (!hasLaunchButton) {
        // Select a model - prefer Qwen3-0.6B-4bit (smallest)
        await modelDropdown.click();
        await page.waitForTimeout(500);

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

      // Launch the instance
      await launchButton.click();

      // Wait for instance to be ready
      await expect(readyIndicator).toBeVisible({ timeout: 300000 }); // 5 min for download
    }

    // Now we should have a running instance - wait for chat input to be available
    await expect(chatInput).toBeVisible({ timeout: 30000 });

    // Type a simple message
    await chatInput.fill("What is 2+2? Reply with just the number.");

    // Click send button
    const sendButton = page.locator('[data-testid="send-button"]');
    await expect(sendButton).toBeVisible();
    await sendButton.click();

    // Wait for user message to appear
    const userMessage = page.locator('[data-testid="user-message"]');
    await expect(userMessage).toBeVisible({ timeout: 10000 });

    // Wait for assistant response
    const assistantMessage = page.locator('[data-testid="assistant-message"]');
    await expect(assistantMessage).toBeVisible({ timeout: 120000 }); // 2 min for response

    // Verify we got some response text
    const responseText = await assistantMessage.textContent();
    expect(responseText).toBeTruthy();
    expect(responseText!.length).toBeGreaterThan(0);
  });
});
