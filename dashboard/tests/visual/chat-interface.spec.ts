import { test, expect } from "@playwright/test";
import { waitForTopologyLoaded } from "../helpers/wait-for-ready";

test.describe("Chat Interface", () => {
  test("should display chat input and send button", async ({ page }) => {
    await page.goto("/");
    await waitForTopologyLoaded(page);

    const chatInput = page.locator('[data-testid="chat-input"]');
    await expect(chatInput).toBeVisible();

    const sendButton = page.locator('[data-testid="send-button"]');
    await expect(sendButton).toBeVisible();
  });

  test("should allow typing in chat input", async ({ page }) => {
    await page.goto("/");
    await waitForTopologyLoaded(page);

    const chatInput = page.locator('[data-testid="chat-input"]');
    await expect(chatInput).toBeVisible();

    await chatInput.fill("Test message");
    await expect(chatInput).toHaveValue("Test message");
  });
});
