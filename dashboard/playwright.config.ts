/// <reference types="node" />
import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: [["html", { open: "never" }], ["list"]],
  use: {
    baseURL: "http://localhost:52415",
    trace: "on-first-retry",
    video: "on",
    screenshot: "only-on-failure",
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
  webServer: {
    command: "cd .. && uv run exo",
    url: "http://localhost:52415/node_id",
    reuseExistingServer: !process.env.CI,
    timeout: 300000, // 5 minutes - CI needs time to install dependencies
    env: {
      ...process.env,
      // Ensure macmon and system tools are accessible
      PATH: `/usr/sbin:/usr/bin:/opt/homebrew/bin:${process.env.PATH}`,
      // Override memory detection for CI (macmon may not work on CI runners)
      // 24GB is typical for GitHub Actions macos-26 runners
      ...(process.env.CI ? { OVERRIDE_MEMORY_MB: "24000" } : {}),
    },
  },
  expect: {
    toHaveScreenshot: {
      maxDiffPixelRatio: 0.05,
      threshold: 0.2,
    },
  },
});
