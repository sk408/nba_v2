# Lightpanda Node API Discovery

This folder provides a direct Node workflow for API discovery through Lightpanda CDP.

## Install

From this folder:

```powershell
npm install playwright-core
```

## Run (Puppeteer, Recommended)

```powershell
node discover-endpoints-puppeteer.mjs --url "https://www.espn.com/nba/game?gameId=401810836"
```

Optional flags:

- `--wsEndpoint "ws://127.0.0.1:9223/"` (default)
- `--waitMs 15000` to capture more late network events
- `--output "../../data/reports/lightpanda_espn_endpoints.json"` to save report

Example:

```powershell
node discover-endpoints-puppeteer.mjs --url "https://www.espn.com/nba/game?gameId=401810836" --waitMs 20000 --output "../../data/reports/espn_game_401810836_endpoints.json"
```

## Notes

- Keep Lightpanda running (`start_lightpanda.bat`).
- This bypasses MCP automation and talks directly to CDP for better reliability when MCP browser actions stall.
- The recommended default endpoint (`9223`) is a dedicated Lightpanda instance to avoid conflicts with Cursor MCP on `9222`.
- `discover-endpoints.mjs` (Playwright-based) is kept for comparison/debugging, but `discover-endpoints-puppeteer.mjs` is currently more reliable with local Lightpanda.
- Root helpers are available:
  - PowerShell: `.\discover_endpoints_lightpanda.ps1 -Url "<url>"`
  - CMD: `discover_endpoints_lightpanda.bat "<url>"`
