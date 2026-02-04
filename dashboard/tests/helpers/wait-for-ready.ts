import { expect, type Page } from "@playwright/test";

const BASE_URL = "http://localhost:52415";

export async function waitForApiReady(
  page: Page,
  timeoutMs = 30000,
): Promise<void> {
  const startTime = Date.now();
  while (Date.now() - startTime < timeoutMs) {
    try {
      const response = await page.request.get(`${BASE_URL}/node_id`);
      if (response.ok()) {
        return;
      }
    } catch {
      // API not ready yet, continue polling
    }
    await page.waitForTimeout(500);
  }
  throw new Error(`API did not become ready within ${timeoutMs}ms`);
}

export async function waitForTopologyLoaded(page: Page): Promise<void> {
  await expect(page.locator('[data-testid="topology-graph"]')).toBeVisible({
    timeout: 30000,
  });
}

export async function waitForModelCards(page: Page): Promise<void> {
  await expect(page.locator('[data-testid="model-card"]').first()).toBeVisible({
    timeout: 30000,
  });
}

export async function selectModelFromLaunchDropdown(
  page: Page,
  modelPattern: RegExp | string,
): Promise<void> {
  // Click the model dropdown in the Launch Instance panel
  const dropdown = page.locator('button:has-text("SELECT MODEL")');
  await expect(dropdown).toBeVisible({ timeout: 30000 });
  await dropdown.click();

  // Wait for dropdown menu to appear and select the model
  const modelOption = page.locator("button").filter({ hasText: modelPattern });
  await expect(modelOption.first()).toBeVisible({ timeout: 10000 });
  await modelOption.first().click();
}

export async function waitForChatReady(page: Page): Promise<void> {
  await expect(page.locator('[data-testid="chat-input"]')).toBeVisible({
    timeout: 10000,
  });
  await expect(page.locator('[data-testid="send-button"]')).toBeVisible({
    timeout: 10000,
  });
}

export async function waitForAssistantMessage(
  page: Page,
  timeoutMs = 60000,
): Promise<void> {
  await expect(
    page.locator('[data-testid="assistant-message"]').last(),
  ).toBeVisible({ timeout: timeoutMs });
}

export async function waitForStreamingComplete(
  page: Page,
  timeoutMs = 120000,
): Promise<void> {
  const startTime = Date.now();
  while (Date.now() - startTime < timeoutMs) {
    const sendButton = page.locator('[data-testid="send-button"]');
    const buttonText = await sendButton.textContent();
    if (
      buttonText &&
      !buttonText.includes("PROCESSING") &&
      !buttonText.includes("...")
    ) {
      return;
    }
    await page.waitForTimeout(500);
  }
  throw new Error(`Streaming did not complete within ${timeoutMs}ms`);
}

export async function selectModel(
  page: Page,
  modelName: string,
): Promise<void> {
  const modelSelector = page.locator('[data-testid="chat-model-selector"]');
  await modelSelector.click();
  await page.locator(`text=${modelName}`).click();
}

export async function sendChatMessage(
  page: Page,
  message: string,
): Promise<void> {
  const chatInput = page.locator('[data-testid="chat-input"]');
  await chatInput.fill(message);
  const sendButton = page.locator('[data-testid="send-button"]');
  await sendButton.click();
}

export async function launchModel(
  page: Page,
  modelCardIndex = 0,
): Promise<void> {
  const modelCards = page.locator('[data-testid="model-card"]');
  const launchButton = modelCards
    .nth(modelCardIndex)
    .locator('[data-testid="launch-button"]');
  await launchButton.click();
}
