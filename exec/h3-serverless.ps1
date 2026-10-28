<#
.SYNOPSIS
  Generic RunPod serverless client: submits the "input" of a JSON job file, polls it, saves outputs.

.DESCRIPTION
  The script knows nothing about workflows or parameters. The job file holds everything:

    {
      "endpointId":     "<runpod endpoint id>",   (or -EndpointId / $env:RUNPOD_ENDPOINT_ID)
      "outputDir":      "C:/out",                  (default: .\output next to the job file)
      "timeoutMinutes": 90,                        (default 90)
      "input":          { ...sent verbatim as the job input... }
    }

  Any string inside "input" that starts with "@" is replaced by the base64 of that file
  (relative paths resolve against the job file's folder). Use forward slashes in JSON paths.

.EXAMPLE
  $env:RUNPOD_API_KEY = "..."
  .\h3-serverless.ps1 .\ref2va_01.json

.EXAMPLE
  # Only print the POST body (e.g. to use with curl); nothing is sent
  .\h3-serverless.ps1 .\ref2va_01.json -BodyOnly > body.json
  .\h3-serverless.ps1 .\ref2va_01.json -BodyOnly -BodyFile body.json
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory, Position = 0)]
    [string]$Config,
    [string]$EndpointId,
    # Print the POST body (JSON) to the output stream and exit: nothing is submitted, no API key needed.
    [switch]$BodyOnly,
    # With -BodyOnly: write the body to this file instead of the output stream.
    [string]$BodyFile
)

$ErrorActionPreference = "Stop"

if (-not $BodyOnly -and -not $env:RUNPOD_API_KEY) { throw "Set the RUNPOD_API_KEY environment variable first." }
if ($BodyFile -and -not $BodyOnly) { throw "-BodyFile requires -BodyOnly." }

$ConfigPath = (Resolve-Path -LiteralPath $Config).Path
$ConfigDir = Split-Path -Parent $ConfigPath
$Cfg = Get-Content -LiteralPath $ConfigPath -Raw -Encoding UTF8 | ConvertFrom-Json

if (-not $EndpointId) { $EndpointId = if ($Cfg.endpointId) { $Cfg.endpointId } else { $env:RUNPOD_ENDPOINT_ID } }
if (-not $BodyOnly -and -not $EndpointId) { throw "No endpoint id: set endpointId in the job file, pass -EndpointId, or set RUNPOD_ENDPOINT_ID." }
if (-not $Cfg.input) { throw "Job file has no 'input' object: $ConfigPath" }

$OutputDir = if ($Cfg.outputDir) { $Cfg.outputDir } else { Join-Path $ConfigDir "output" }
$TimeoutMinutes = if ($Cfg.timeoutMinutes) { [int]$Cfg.timeoutMinutes } else { 90 }

# Replace "@path" strings with file contents (base64), in place so arrays keep their shape.
function Get-FileBase64([string]$Value) {
    $Path = $Value.Substring(1)
    if (-not [System.IO.Path]::IsPathRooted($Path)) { $Path = Join-Path $ConfigDir $Path }
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "File not found: $Path" }
    $Bytes = [System.IO.File]::ReadAllBytes($Path)
    Write-Host ("{0}: {1:N2} MB" -f $Path, ($Bytes.Length / 1MB))
    return [Convert]::ToBase64String($Bytes)
}

function Test-FileRef($Value) { return ($Value -is [string]) -and $Value.StartsWith("@") }

function Expand-FileRefs($Node) {
    if ($Node -is [System.Array]) {
        for ($i = 0; $i -lt $Node.Count; $i++) {
            if (Test-FileRef $Node[$i]) { $Node[$i] = Get-FileBase64 $Node[$i] }
            else { Expand-FileRefs $Node[$i] }
        }
    }
    elseif ($Node -is [System.Management.Automation.PSCustomObject]) {
        foreach ($Prop in $Node.PSObject.Properties) {
            if (Test-FileRef $Prop.Value) { $Prop.Value = Get-FileBase64 $Prop.Value }
            else { Expand-FileRefs $Prop.Value }
        }
    }
}

Expand-FileRefs $Cfg.input

$Headers = @{
    Authorization  = "Bearer $($env:RUNPOD_API_KEY)"
    "Content-Type" = "application/json"
}

$Payload = @{ input = $Cfg.input } | ConvertTo-Json -Depth 20 -Compress

if ($BodyOnly) {
    if ($BodyFile) {
        [System.IO.File]::WriteAllText($BodyFile, $Payload, (New-Object System.Text.UTF8Encoding($false)))
        Write-Host ("Wrote {0:N2} MB body to {1}" -f ($Payload.Length / 1MB), $BodyFile)
    }
    else {
        $Payload
    }
    return
}

Write-Host ("Request body: {0:N2} MB" -f ($Payload.Length / 1MB))

# ---- Submit ----
$Job = Invoke-RestMethod -Method Post -Uri "https://api.runpod.ai/v2/$EndpointId/run" `
    -Headers $Headers -Body $Payload -TimeoutSec 180
if (-not $Job.id) {
    $Job | ConvertTo-Json -Depth 10
    throw "RunPod did not return a job ID."
}
$JobId = $Job.id
Write-Host "Job ID: $JobId  Status: $($Job.status)"

# ---- Poll ----
$StatusUrl = "https://api.runpod.ai/v2/$EndpointId/status/$JobId"
$Deadline = (Get-Date).AddMinutes($TimeoutMinutes)
$Result = $null
while ((Get-Date) -lt $Deadline) {
    Start-Sleep -Seconds 5
    $Result = Invoke-RestMethod -Method Get -Uri $StatusUrl -Headers $Headers -TimeoutSec 60
    Write-Host "$(Get-Date -Format 'HH:mm:ss')  Status: $($Result.status)"
    if ($Result.status -eq "COMPLETED") { break }
    if ($Result.status -in @("FAILED", "CANCELLED", "TIMED_OUT")) {
        $Result | ConvertTo-Json -Depth 20
        throw "RunPod job ended with status: $($Result.status)"
    }
}
if ($Result.status -ne "COMPLETED") { throw "Timed out waiting for job $JobId." }

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$JsonPath = Join-Path $OutputDir "$JobId-result.json"
$Result | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath $JsonPath -Encoding UTF8

# The handler reports invalid input as output.error (job still COMPLETED).
if ($Result.output.error) { throw "Handler error: $($Result.output.error)" }
foreach ($Warning in @($Result.output.warnings)) { if ($Warning) { Write-Warning $Warning } }
if ($null -ne $Result.output.seed) { Write-Host "Seed: $($Result.output.seed)" }
if ($Result.output.provisioned) {
    Write-Host "Provisioned:"
    $Result.output.provisioned | ForEach-Object { Write-Host "  $_" }
}

# ---- Save outputs ----
foreach ($Output in @($Result.output.outputs)) {
    if (-not $Output) { continue }
    $FileName = [System.IO.Path]::GetFileName([string]$Output.filename)
    if (-not $FileName) { $FileName = "output.bin" }
    $Destination = Join-Path $OutputDir $FileName

    if ($Output.url) {
        Invoke-WebRequest -Uri $Output.url -OutFile $Destination -TimeoutSec 600
    }
    elseif ($Output.b64) {
        [System.IO.File]::WriteAllBytes($Destination, [Convert]::FromBase64String([string]$Output.b64))
    }
    else {
        Write-Warning "No URL or Base64 payload for output: $FileName"
        continue
    }
    Write-Host "Saved: $Destination"
}
Write-Host "Finished. Result: $JsonPath"
