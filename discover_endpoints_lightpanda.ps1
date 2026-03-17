param(
    [Parameter(Mandatory = $true)]
    [string]$Url,
    [string]$Output = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$nodeToolDir = Join-Path $repoRoot "tools/lightpanda-node"
$nodeContainerName = "lightpanda-node"
$nodeImage = "lightpanda/browser:nightly"
$nodeCdpPort = 9223
$nodeCdpEndpoint = "ws://127.0.0.1:$nodeCdpPort/"

if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    Write-Error "[lightpanda-node] Node.js not found. Install Node 18+ first."
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Error "[lightpanda-node] Docker CLI not found. Install Docker Desktop first."
}

docker info *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Error "[lightpanda-node] Docker daemon is not running. Start Docker Desktop and retry."
}

if (-not (Test-Path (Join-Path $nodeToolDir "package.json"))) {
    Write-Error "[lightpanda-node] Missing tools/lightpanda-node/package.json"
}

if (-not (Test-Path (Join-Path $nodeToolDir "node_modules/puppeteer-core"))) {
    Write-Host "[lightpanda-node] Installing node dependencies..."
    Push-Location $nodeToolDir
    npm install
    $installExit = $LASTEXITCODE
    Pop-Location
    if ($installExit -ne 0) {
        Write-Error "[lightpanda-node] npm install failed."
    }
}

docker inspect $nodeContainerName *> $null
if ($LASTEXITCODE -eq 0) {
    docker rm -f $nodeContainerName *> $null
}

Write-Host "[lightpanda-node] Starting dedicated Lightpanda container on port $nodeCdpPort..."
docker run -d --name $nodeContainerName -p ${nodeCdpPort}:9222 $nodeImage /bin/lightpanda serve --host 0.0.0.0 --port 9222 --log_level info --timeout 3600 *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Error "[lightpanda-node] Failed to start $nodeContainerName container."
}

$nodeArgs = @(
    "discover-endpoints-puppeteer.mjs",
    "--url", $Url,
    "--wsEndpoint", $nodeCdpEndpoint
)

if ($Output) {
    $resolvedOutput = if ([System.IO.Path]::IsPathRooted($Output)) {
        $Output
    } else {
        Join-Path $repoRoot $Output
    }
    $outputDir = Split-Path -Parent $resolvedOutput
    if ($outputDir -and -not (Test-Path $outputDir)) {
        New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
    }
    $nodeArgs += @("--output", $resolvedOutput)
}

Push-Location $nodeToolDir
& node @nodeArgs
$nodeExit = $LASTEXITCODE
Pop-Location

exit $nodeExit
