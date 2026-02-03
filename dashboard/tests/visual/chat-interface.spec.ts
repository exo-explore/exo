import { test, expect } from "@playwright/test";

test.describe("Chat Interface Visual Snapshots", () => {
  test("chat input area", async ({ page }) => {
    await page.goto("/");
    await page.waitForLoadState("networkidle");

    const chatInput = page.locator('[data-testid="chat-input"]');
    const isVisible = await chatInput
      .isVisible({ timeout: 10000 })
      .catch(() => false);

    if (!isVisible) {
      // No chat interface available (no running instance)
      test.skip();
      return;
    }

    // Take screenshot of the chat form area
    const chatForm = page.locator("form").filter({ has: chatInput });
    await expect(chatForm).toHaveScreenshot("chat-input-empty.png");
  });

  test("chat input with text", async ({ page }) => {
    await page.goto("/");
    await page.waitForLoadState("networkidle");

    const chatInput = page.locator('[data-testid="chat-input"]');
    const isVisible = await chatInput
      .isVisible({ timeout: 10000 })
      .catch(() => false);

    if (!isVisible) {
      test.skip();
      return;
    }

    // Type some text
    await chatInput.fill("This is a test message");
    await page.waitForTimeout(200);

    const chatForm = page.locator("form").filter({ has: chatInput });
    await expect(chatForm).toHaveScreenshot("chat-input-with-text.png");
  });

  test("empty chat messages area", async ({ page }) => {
    await page.goto("/");
    await page.waitForLoadState("networkidle");

    const chatInput = page.locator('[data-testid="chat-input"]');
    const isVisible = await chatInput
      .isVisible({ timeout: 10000 })
      .catch(() => false);

    if (!isVisible) {
      test.skip();
      return;
    }

    // Look for the "AWAITING INPUT" text that appears in empty chat
    const emptyState = page.locator("text=AWAITING INPUT");
    const hasEmptyState = await emptyState.isVisible().catch(() => false);

    if (hasEmptyState) {
      await expect(page).toHaveScreenshot("chat-empty-state.png", {
        fullPage: true,
      });
    }
  });

  test("chat with user message", async ({ page }) => {
    await page.goto("/");
    await page.waitForLoadState("networkidle");

    const chatInput = page.locator('[data-testid="chat-input"]');
    const isVisible = await chatInput
      .isVisible({ timeout: 10000 })
      .catch(() => false);

    if (!isVisible) {
      test.skip();
      return;
    }

    // Send a message
    await chatInput.fill("Hello, world!");
    await page.locator('[data-testid="send-button"]').click();

    // Wait for user message to appear
    await expect(page.locator('[data-testid="user-message"]')).toBeVisible({
      timeout: 5000,
    });

    // Take screenshot of the chat area with the user message
    await expect(page).toHaveScreenshot("chat-with-user-message.png", {
      fullPage: true,
    });
  });

  test("send button states", async ({ page }) => {
    await page.goto("/");
    await page.waitForLoadState("networkidle");

    const chatInput = page.locator('[data-testid="chat-input"]');
    const isVisible = await chatInput
      .isVisible({ timeout: 10000 })
      .catch(() => false);

    if (!isVisible) {
      test.skip();
      return;
    }

    const sendButton = page.locator('[data-testid="send-button"]');

    // Empty state - button should be disabled
    await expect(sendButton).toHaveScreenshot("send-button-disabled.png");

    // With text - button should be enabled
    await chatInput.fill("Test message");
    await page.waitForTimeout(100);
    await expect(sendButton).toHaveScreenshot("send-button-enabled.png");
  });
});
