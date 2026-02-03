import { test, expect } from "@playwright/test";
import { waitForTopologyLoaded } from "../helpers/wait-for-ready";

test.describe("Chat Interface Visual Snapshots", () => {
  test("chat input area", async ({ page }) => {
    await page.goto("/");
    await waitForTopologyLoaded(page);

    const chatInput = page.locator('[data-testid="chat-input"]');
    await expect(chatInput).toBeVisible({ timeout: 10000 });

    // Take screenshot of the chat form area
    const chatForm = page.locator("form").filter({ has: chatInput });
    await expect(chatForm).toHaveScreenshot("chat-input-empty.png");
  });

  test("chat input with text", async ({ page }) => {
    await page.goto("/");
    await waitForTopologyLoaded(page);

    const chatInput = page.locator('[data-testid="chat-input"]');
    await expect(chatInput).toBeVisible({ timeout: 10000 });

    // Type some text
    await chatInput.fill("This is a test message");
    await page.waitForTimeout(200);

    const chatForm = page.locator("form").filter({ has: chatInput });
    await expect(chatForm).toHaveScreenshot("chat-input-with-text.png");
  });

  test("send button states", async ({ page }) => {
    await page.goto("/");
    await waitForTopologyLoaded(page);

    const chatInput = page.locator('[data-testid="chat-input"]');
    await expect(chatInput).toBeVisible({ timeout: 10000 });

    const sendButton = page.locator('[data-testid="send-button"]');

    // Empty state - button should be disabled
    await expect(sendButton).toHaveScreenshot("send-button-disabled.png");

    // With text - button should be enabled
    await chatInput.fill("Test message");
    await page.waitForTimeout(100);
    await expect(sendButton).toHaveScreenshot("send-button-enabled.png");
  });
});
