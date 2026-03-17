# Lightpanda on Windows (Node First)

This workspace now uses a node-first Lightpanda flow for web/API discovery.  
Project MCP wiring was intentionally removed to avoid `action: "list"` style hangs.

## What Was Added

- Rule: `.cursor/rules/lightpanda-web-requests.mdc` (always applies).
- Root helpers:
  - `discover_endpoints_lightpanda.ps1` (recommended)
  - `discover_endpoints_lightpanda.bat`
- Node tooling:
  - `tools/lightpanda-node/discover-endpoints-puppeteer.mjs`
  - `tools/lightpanda-node/package.json`

## Prerequisites

- Docker Desktop running.
- Node.js 18+ installed.
- Internet access for image/dependency fetches.

## Quick Start

PowerShell:

```powershell
.\discover_endpoints_lightpanda.ps1 -Url "https://target.site/path"
```

With output path:

```powershell
.\discover_endpoints_lightpanda.ps1 -Url "https://target.site/path" -Output "data/reports/target_report.json"
```

CMD:

```cmd
discover_endpoints_lightpanda.bat "https://target.site/path" "data/reports/target_report.json"
```

If using `cmd.exe` and URL contains `&`, escape it as `^&`.

## What The Helper Does

- Ensures node dependencies are installed.
- Recreates a dedicated `lightpanda-node` container each run.
- Uses Lightpanda CDP on `ws://127.0.0.1:9223/`.
- Captures discovered network/API endpoints in structured JSON.

## Verification

Check helper container:

```powershell
docker ps --filter "name=lightpanda-node"
```

Run a smoke test:

```powershell
.\discover_endpoints_lightpanda.ps1 -Url "https://example.com"
```

Expected: JSON output with at least one endpoint.

## Troubleshooting

- `Navigating frame was detached` can happen on dynamic apps. If the report includes endpoints, treat as partial success.
- If the report is empty, retry once; helper recreates a clean container each run.
- If docker is unavailable, use the WSL installer script `setup_lightpanda_wsl.bat` and run Lightpanda manually in WSL for advanced fallback.
