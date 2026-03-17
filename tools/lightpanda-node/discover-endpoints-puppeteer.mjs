import fs from "node:fs/promises";
import puppeteer from "puppeteer-core";

const DEFAULT_WS_ENDPOINT = "ws://127.0.0.1:9223/";
const DEFAULT_WAIT_MS = 12_000;

function parseArgs(argv) {
  const args = {
    url: "",
    wsEndpoint: DEFAULT_WS_ENDPOINT,
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
    if (token === "--wsEndpoint") {
      args.wsEndpoint = argv[i + 1] ?? DEFAULT_WS_ENDPOINT;
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
  const known = ["authorization", "cookie", "x-api-key", "x-auth-token", "x-csrf-token"];
  for (const key of known) {
    if (headers[key]) {
      clues.push(key);
    }
  }
  return clues;
}

function formatError(error) {
  if (error instanceof Error) {
    return `${error.message}${error.stack ? `\n${error.stack}` : ""}`;
  }
  try {
    return JSON.stringify(error, null, 2);
  } catch {
    return String(error);
  }
}

async function run() {
  const { url, wsEndpoint, waitMs, output } = parseArgs(process.argv.slice(2));

  let browser;
  let page;
  let navigationError = "";
  const events = [];
  const started = Date.now();
  let stage = "init";

  try {
    try {
      stage = "connect";
      browser = await puppeteer.connect({
        browserWSEndpoint: wsEndpoint,
        protocolTimeout: 45_000,
      });

      stage = "newPage";
      page = await browser.newPage();
    } catch (error) {
      throw new Error(`[${stage}] ${formatError(error)}`);
    }

    page.on("request", (request) => {
      const headers = request.headers();
      events.push({
        type: "request",
        ts: Date.now(),
        method: request.method(),
        url: request.url(),
        authHeaderHints: authCluesFromHeaders(headers),
        postDataPreview: request.postData()?.slice(0, 300) ?? "",
      });
    });

    page.on("response", async (response) => {
      const request = response.request();
      const headers = response.headers();
      events.push({
        type: "response",
        ts: Date.now(),
        method: request.method(),
        url: response.url(),
        status: response.status(),
        contentType: headers["content-type"] ?? "",
        authHeaderHints: authCluesFromHeaders(headers),
      });
    });

    try {
      stage = "goto";
      await page.goto(url, { waitUntil: "domcontentloaded", timeout: 45_000 });
    } catch (error) {
      navigationError = error instanceof Error ? error.message : formatError(error);
      console.warn(`Navigation warning: ${navigationError}`);
    }

    await new Promise((resolve) => setTimeout(resolve, waitMs));

    const endpointMap = new Map();
    for (const row of events) {
      const key = `${row.method} ${row.url}`;
      if (!endpointMap.has(key)) {
        endpointMap.set(key, {
          method: row.method,
          url: row.url,
          statuses: new Set(),
          contentTypes: new Set(),
          authClues: new Set(),
          bodySamples: [],
        });
      }
      const slot = endpointMap.get(key);
      if (row.status) slot.statuses.add(row.status);
      if (row.contentType) slot.contentTypes.add(row.contentType);
      for (const clue of row.authHeaderHints) slot.authClues.add(clue);
      if (row.postDataPreview && slot.bodySamples.length < 2) {
        slot.bodySamples.push(row.postDataPreview);
      }
    }

    const endpoints = Array.from(endpointMap.values()).map((row) => ({
      method: row.method,
      url: row.url,
      statuses: Array.from(row.statuses),
      contentTypes: Array.from(row.contentTypes),
      authClues: Array.from(row.authClues),
      bodySamples: row.bodySamples,
    }));

    const report = {
      targetUrl: url,
      wsEndpoint,
      elapsedMs: Date.now() - started,
      navigationError,
      totalEvents: events.length,
      uniqueEndpoints: endpoints.length,
      endpoints,
    };

    if (output) {
      await fs.writeFile(output, JSON.stringify(report, null, 2), "utf8");
      console.log(`Wrote endpoint report to ${output}`);
    } else {
      console.log(JSON.stringify(report, null, 2));
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
        await browser.disconnect();
      } catch {
        // best effort close
      }
    }
  }
}

run().catch((error) => {
  console.error(`discover-endpoints-puppeteer failed: ${formatError(error)}`);
  process.exitCode = 1;
});
