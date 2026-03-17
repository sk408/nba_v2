import fs from "node:fs/promises";
import { chromium } from "playwright-core";

const DEFAULT_CDP_ENDPOINT = "ws://127.0.0.1:9222";
const DEFAULT_WAIT_MS = 12_000;
const NAV_TIMEOUT_MS = 60_000;

function parseArgs(argv) {
  const args = {
    url: "",
    cdpEndpoint: DEFAULT_CDP_ENDPOINT,
    waitMs: DEFAULT_WAIT_MS,
    output: "",
  };

  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (token === "--url") {
      args.url = argv[i + 1] ?? "";
      i += 1;
      continue;
    }
    if (token === "--cdpEndpoint") {
      args.cdpEndpoint = argv[i + 1] ?? DEFAULT_CDP_ENDPOINT;
      i += 1;
      continue;
    }
    if (token === "--waitMs") {
      const parsed = Number(argv[i + 1]);
      args.waitMs = Number.isFinite(parsed) && parsed >= 0 ? parsed : DEFAULT_WAIT_MS;
      i += 1;
      continue;
    }
    if (token === "--output") {
      args.output = argv[i + 1] ?? "";
      i += 1;
    }
  }

  if (!args.url) {
    throw new Error("Missing --url argument");
  }

  return args;
}

function authCluesFromHeaders(headers) {
  const clues = [];
  const known = [
    "authorization",
    "cookie",
    "x-api-key",
    "x-auth-token",
    "x-csrf-token",
    "x-espn-authorization",
  ];
  for (const key of known) {
    if (headers[key]) {
      clues.push(key);
    }
  }
  return clues;
}

async function waitForQuietNetwork(getLastActivity, { quietMs = 2_500, maxMs = 20_000 } = {}) {
  const started = Date.now();
  while (Date.now() - started < maxMs) {
    if (Date.now() - getLastActivity() >= quietMs) {
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
}

async function run() {
  const { url, cdpEndpoint, waitMs, output } = parseArgs(process.argv.slice(2));

  const discovered = [];
  const byKey = new Map();
  let lastActivity = Date.now();
  const startedAt = Date.now();
  let browser;
  let context;
  let page;

  try {
    browser = await chromium.connectOverCDP(cdpEndpoint, { timeout: 30_000 });

    context = browser.contexts()[0];
    if (!context) {
      context = await browser.newContext();
    }

    page = context.pages()[0];
    if (!page) {
      page = await context.newPage();
    }

    page.setDefaultTimeout(NAV_TIMEOUT_MS);
    page.setDefaultNavigationTimeout(NAV_TIMEOUT_MS);

    page.on("request", (request) => {
      const headers = request.headers();
      const entry = {
        type: "request",
        ts: Date.now(),
        method: request.method(),
        url: request.url(),
        postDataPreview: request.postData()?.slice(0, 300) ?? "",
        authHeaderHints: authCluesFromHeaders(headers),
      };
      discovered.push(entry);
      lastActivity = Date.now();
    });

    page.on("response", async (response) => {
      const req = response.request();
      const headers = response.headers();
      const entry = {
        type: "response",
        ts: Date.now(),
        method: req.method(),
        url: response.url(),
        status: response.status(),
        contentType: headers["content-type"] ?? "",
        authHeaderHints: authCluesFromHeaders(headers),
      };
      discovered.push(entry);
      lastActivity = Date.now();
    });

    let navigationError = "";
    try {
      await page.goto(url, { waitUntil: "commit", timeout: 45_000 });
    } catch (error) {
      navigationError = error instanceof Error ? error.message : String(error);
      console.warn(`Navigation warning: ${navigationError}`);
    }
    await page.waitForTimeout(waitMs);
    await waitForQuietNetwork(() => lastActivity);

    const endpointMap = new Map();
    for (const row of discovered) {
      const key = `${row.method} ${row.url}`;
      if (!endpointMap.has(key)) {
        endpointMap.set(key, {
          method: row.method,
          url: row.url,
          firstSeenType: row.type,
          statuses: new Set(),
          contentTypes: new Set(),
          authClues: new Set(),
          samples: [],
        });
      }
      const slot = endpointMap.get(key);
      if (row.status) slot.statuses.add(row.status);
      if (row.contentType) slot.contentTypes.add(row.contentType);
      for (const clue of row.authHeaderHints) slot.authClues.add(clue);
      if (row.postDataPreview && slot.samples.length < 2) {
        slot.samples.push(row.postDataPreview);
      }
    }

    const endpoints = Array.from(endpointMap.values())
      .map((item) => ({
        method: item.method,
        url: item.url,
        statuses: Array.from(item.statuses),
        contentTypes: Array.from(item.contentTypes),
        authClues: Array.from(item.authClues),
        bodySamples: item.samples,
      }))
      .sort((a, b) => a.url.localeCompare(b.url));

    for (const endpoint of endpoints) {
      const key = `${endpoint.method} ${endpoint.url}`;
      byKey.set(key, (byKey.get(key) ?? 0) + 1);
    }

    const summary = {
      targetUrl: url,
      cdpEndpoint,
      elapsedMs: Date.now() - startedAt,
      navigationError,
      totalEvents: discovered.length,
      uniqueEndpoints: endpoints.length,
      endpoints,
      eventCounts: Array.from(byKey.entries()).map(([key, count]) => ({ key, count })),
    };

    if (output) {
      await fs.writeFile(output, JSON.stringify(summary, null, 2), "utf8");
      console.log(`Wrote endpoint report to ${output}`);
    } else {
      console.log(JSON.stringify(summary, null, 2));
    }
  } finally {
    if (page) {
      try {
        await page.close();
      } catch {
        // best effort close
      }
    }
    if (browser) {
      try {
        await browser.close();
      } catch {
        // best effort close
      }
    }
  }
}

run().catch((error) => {
  console.error(`discover-endpoints failed: ${error instanceof Error ? error.message : String(error)}`);
  process.exitCode = 1;
});
