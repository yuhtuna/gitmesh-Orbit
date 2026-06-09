<#
.SYNOPSIS
    Onboard an external GitLab project to a deployed GitMesh engine (multi-project).

    EXPERIMENTAL / NIGHTLY — under development, NOT production-tested. Use only
    for trying out cross-project onboarding. The main single-project workflow
    does not depend on this script.

.DESCRIPTION
    Self-serve onboarding: point any GitLab project at an already-deployed GitMesh
    engine WITHOUT cloning this repo per project. This script:
      1. Validates that the supplied access token can reach the target project.
      2. Registers the project's config (URL + API token + webhook secret) in the
         central Modal registry so compute stages can post results back to it.
      3. Creates/updates an Issues webhook on the target project that points at the
         engine's deployed webhook endpoint.

    After onboarding, opening an issue titled "MeshGen: <prompt>" in the target
    project triggers the engine pipeline, and all comments / uploads / auto-close
    are routed back to that project.

.PARAMETER ProjectUrl
    Full URL or path of the target project (e.g. https://gitlab.com/group/repo
    or "group/repo"). Used to resolve the numeric project id.

.PARAMETER AccessToken
    A GitLab access token for the target project with 'api' scope. Stored in the
    central Modal registry and used by the engine to post results back.

.PARAMETER WebhookUrl
    The engine's deployed webhook endpoint (the gitlab_issue_listener URL printed
    by setup_remote.ps1 / modal deploy gitlab_webhook.py).

.PARAMETER WebhookSecret
    Shared secret sent as the x-gitlab-token header and validated by the engine.

.PARAMETER GitLabBaseUrl
    GitLab base URL. Defaults to https://gitlab.com.

.PARAMETER TriggerRef
    Git ref the engine pipeline runs on. Defaults to main.

.PARAMETER AutoClose
    Whether the engine should auto-close the issue when done. Defaults to true.

.EXAMPLE
    .\onboard_project.ps1 -ProjectUrl "group/repo" -AccessToken "glpat-xxx" `
        -WebhookUrl "https://xxx.modal.run" -WebhookSecret "shared-secret"
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string]$ProjectUrl,
    [Parameter(Mandatory = $true)] [string]$AccessToken,
    [Parameter(Mandatory = $true)] [string]$WebhookUrl,
    [Parameter(Mandatory = $true)] [string]$WebhookSecret,
    [string]$GitLabBaseUrl = "https://gitlab.com",
    [string]$TriggerRef = "main",
    [ValidateSet("true", "false")] [string]$AutoClose = "true"
)

$ErrorActionPreference = "Stop"

function Get-ModalCommand {
    foreach ($p in @(".\.venv\Scripts\modal.exe", ".\.venv\bin\modal", ".\venv\Scripts\modal.exe", ".\venv\bin\modal")) {
        if (Test-Path $p) { return $p }
    }
    return "modal"
}

function Resolve-ProjectPath {
    param([string]$Url, [string]$BaseUrl)

    $value = $Url.Trim()
    if ($value -match '^https?://') {
        $u = [Uri]$value
        return $u.AbsolutePath.Trim('/')
    }
    return $value.Trim('/')
}

$baseUrl = $GitLabBaseUrl.TrimEnd('/')
$headers = @{ "PRIVATE-TOKEN" = $AccessToken }
$projectPath = Resolve-ProjectPath -Url $ProjectUrl -BaseUrl $baseUrl
$projectEscaped = [Uri]::EscapeDataString($projectPath)

Write-Host "==> Validating access to target project: $projectPath" -ForegroundColor Cyan
try {
    $project = Invoke-RestMethod -Method Get -Uri "$baseUrl/api/v4/projects/$projectEscaped" -Headers $headers -ErrorAction Stop
} catch {
    throw "Could not access project '$projectPath'. Check the URL and that the token has 'api' scope. $($_.Exception.Message)"
}

$projectId = "$($project.id)"
Write-Host "[+] Resolved project id: $projectId ($($project.path_with_namespace))" -ForegroundColor Green

Write-Host "==> Registering project in the GitMesh registry..." -ForegroundColor Cyan
$modal = Get-ModalCommand
& $modal run project_registry.py --action register `
    --project-id $projectId `
    --gitlab-url $baseUrl `
    --api-token $AccessToken `
    --webhook-secret $WebhookSecret `
    --trigger-ref $TriggerRef `
    --auto-close $AutoClose
if ($LASTEXITCODE -ne 0) {
    throw "Failed to register project in the Modal registry (exit code $LASTEXITCODE)."
}
Write-Host "[+] Project registered in the central registry." -ForegroundColor Green

Write-Host "==> Configuring Issues webhook on the target project..." -ForegroundColor Cyan
$hooksUrl = "$baseUrl/api/v4/projects/$projectEscaped/hooks"
$hooks = Invoke-RestMethod -Method Get -Uri $hooksUrl -Headers $headers -ErrorAction Stop

$existing = $null
foreach ($hook in @($hooks)) {
    if ($hook.url -eq $WebhookUrl) { $existing = $hook; break }
}

$body = @{
    url = $WebhookUrl
    token = $WebhookSecret
    issues_events = $true
    push_events = $false
    merge_requests_events = $false
    tag_push_events = $false
    note_events = $false
    enable_ssl_verification = $true
}

if ($null -ne $existing) {
    $null = Invoke-RestMethod -Method Put -Uri "$hooksUrl/$($existing.id)" -Headers $headers -Body $body -ErrorAction Stop
    Write-Host "[+] Updated existing Issues webhook." -ForegroundColor Green
} else {
    $null = Invoke-RestMethod -Method Post -Uri $hooksUrl -Headers $headers -Body $body -ErrorAction Stop
    Write-Host "[+] Created Issues webhook." -ForegroundColor Green
}

Write-Host ""
Write-Host "Onboarding complete." -ForegroundColor Green
Write-Host "Open an issue titled 'MeshGen: <your prompt>' in $($project.path_with_namespace) to run the pipeline." -ForegroundColor Green
