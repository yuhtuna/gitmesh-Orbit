# ==============================================================================
# GitMesh Remote Setup Bootstrap (PowerShell)
# Reads a local .env file, validates required values, creates the Modal secret,
# deploys the Modal apps, and prints the remaining GitLab setup values.
# ==============================================================================

[CmdletBinding()]
param(
    [string]$EnvFile = ".env",
    [string]$WebhookUrl = "",
    [switch]$ProtectSensitiveVars,
    [switch]$DryRun,
    [switch]$SkipGitLabApi,
    [switch]$SkipDeploy,
    [switch]$SkipSecret
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Get-EnvMap {
    param([string]$Path)

    if (-not (Test-Path $Path)) {
        throw "Environment file not found: $Path"
    }

    $values = @{}
    foreach ($rawLine in Get-Content $Path) {
        $line = $rawLine.Trim()
        if (-not $line -or $line.StartsWith("#")) {
            continue
        }
        $parts = $line -split '=', 2
        if ($parts.Count -ne 2) {
            continue
        }
        $key = $parts[0].Trim()
        $value = $parts[1].Trim()
        if (($value.StartsWith('"') -and $value.EndsWith('"')) -or ($value.StartsWith("'") -and $value.EndsWith("'"))) {
            $value = $value.Substring(1, $value.Length - 2)
        }
        $values[$key] = $value
    }

    return $values
}

function Require-Value {
    param(
        [hashtable]$Config,
        [string]$Name
    )

    if (-not $Config.ContainsKey($Name) -or [string]::IsNullOrWhiteSpace($Config[$Name])) {
        throw "Missing required value in .env: $Name"
    }
}

function Get-ModalCommand {
    if (Test-Path ".\.venv\Scripts\modal.exe") {
        return ".\.venv\Scripts\modal.exe"
    }
    if (Test-Path ".\.venv\bin\modal") {
        return ".\.venv\bin\modal"
    }
    if (Test-Path ".\venv\Scripts\modal.exe") {
        return ".\venv\Scripts\modal.exe"
    }
    if (Test-Path ".\venv\bin\modal") {
        return ".\venv\bin\modal"
    }
    return "modal"
}

function Require-Command {
    param([string]$Name)

    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command not found in PATH: $Name"
    }
}

function Resolve-GitLabBaseUrl {
    param([hashtable]$Config)

    if ($Config.ContainsKey("GITLAB_URL") -and -not [string]::IsNullOrWhiteSpace($Config["GITLAB_URL"])) {
        return $Config["GITLAB_URL"].TrimEnd('/')
    }
    return "https://gitlab.com"
}

function Get-GitLabApiHeaders {
    param([hashtable]$Config)

    return @{
        "PRIVATE-TOKEN" = $Config["GITLAB_API_TOKEN"]
    }
}

function Invoke-GitLabApi {
    param(
        [string]$Method,
        [string]$Uri,
        [hashtable]$Headers,
        [object]$Body
    )

    $params = @{
        Method = $Method
        Uri = $Uri
        Headers = $Headers
        ErrorAction = "Stop"
    }

    if ($null -ne $Body) {
        $params["Body"] = $Body
        $params["ContentType"] = "application/x-www-form-urlencoded"
    }

    try {
        return Invoke-RestMethod @params
    } catch {
        $statusCode = "unknown"
        if ($_.Exception.Response -and $_.Exception.Response.StatusCode) {
            $statusCode = [int]$_.Exception.Response.StatusCode
        }
        throw "GitLab API $Method $Uri failed (HTTP $statusCode): $($_.Exception.Message)"
    }
}

function Set-GitLabVariable {
    param(
        [string]$BaseUrl,
        [string]$ProjectId,
        [hashtable]$Headers,
        [string]$Key,
        [string]$Value,
        [bool]$Masked,
        [bool]$Protected
    )

    $projectEscaped = [Uri]::EscapeDataString($ProjectId)
    $keyEscaped = [Uri]::EscapeDataString($Key)
    $getUrl = "$BaseUrl/api/v4/projects/$projectEscaped/variables/$keyEscaped"

    $exists = $false
    try {
        $null = Invoke-RestMethod -Method Get -Uri $getUrl -Headers $Headers -ErrorAction Stop
        $exists = $true
    } catch {
        $statusCode = $null
        if ($_.Exception.Response -and $_.Exception.Response.StatusCode) {
            $statusCode = [int]$_.Exception.Response.StatusCode
        }
        if ($statusCode -ne 404) {
            throw "Failed checking GitLab variable '$Key': $($_.Exception.Message)"
        }
    }

    $body = @{
        value = $Value
        masked = $Masked
        protected = $Protected
    }

    if ($exists) {
        $null = Invoke-GitLabApi -Method Put -Uri $getUrl -Headers $Headers -Body $body
        Write-Host "[+] Updated GitLab CI variable: $Key" -ForegroundColor Green
    } else {
        $createUrl = "$BaseUrl/api/v4/projects/$projectEscaped/variables"
        $createBody = @{
            key = $Key
            value = $Value
            masked = $Masked
            protected = $Protected
        }
        $null = Invoke-GitLabApi -Method Post -Uri $createUrl -Headers $Headers -Body $createBody
        Write-Host "[+] Created GitLab CI variable: $Key" -ForegroundColor Green
    }
}

function Upsert-GitLabWebhook {
    param(
        [string]$BaseUrl,
        [string]$ProjectId,
        [hashtable]$Headers,
        [string]$HookUrl,
        [string]$SecretToken
    )

    $projectEscaped = [Uri]::EscapeDataString($ProjectId)
    $hooksUrl = "$BaseUrl/api/v4/projects/$projectEscaped/hooks"
    $hooks = Invoke-GitLabApi -Method Get -Uri $hooksUrl -Headers $Headers -Body $null

    $existing = $null
    foreach ($hook in @($hooks)) {
        if ($hook.url -eq $HookUrl) {
            $existing = $hook
            break
        }
    }

    $body = @{
        url = $HookUrl
        token = $SecretToken
        issues_events = $true
        push_events = $false
        merge_requests_events = $false
        tag_push_events = $false
        note_events = $false
        enable_ssl_verification = $true
    }

    if ($null -ne $existing) {
        $updateUrl = "$hooksUrl/$($existing.id)"
        $null = Invoke-GitLabApi -Method Put -Uri $updateUrl -Headers $Headers -Body $body
        Write-Host "[+] Updated existing GitLab webhook for Issue events." -ForegroundColor Green
    } else {
        $null = Invoke-GitLabApi -Method Post -Uri $hooksUrl -Headers $Headers -Body $body
        Write-Host "[+] Created GitLab webhook for Issue events." -ForegroundColor Green
    }
}

function Get-UrlFromText {
    param([string]$Text)

    if ([string]::IsNullOrWhiteSpace($Text)) {
        return ""
    }

    $matches = [regex]::Matches($Text, 'https://[^\s"'']+')
    foreach ($m in $matches) {
        $candidate = $m.Value.TrimEnd('.', ',', ';', ')')
        if ($candidate -match 'modal\.run|modal\.site|modal\.com') {
            return $candidate
        }
    }

    if ($matches.Count -gt 0) {
        return $matches[0].Value.TrimEnd('.', ',', ';', ')')
    }

    return ""
}

$config = Get-EnvMap -Path $EnvFile

foreach ($requiredKey in @(
    "GCP_PROJECT_ID",
    "GCP_SERVICE_ACCOUNT_JSON",
    "GITLAB_PROJECT_ID",
    "GITLAB_TRIGGER_TOKEN",
    "GITLAB_WEBHOOK_SECRET",
    "MODAL_TOKEN_ID",
    "MODAL_TOKEN_SECRET"
)) {
    Require-Value -Config $config -Name $requiredKey
}

if (-not $SkipGitLabApi) {
    Require-Value -Config $config -Name "GITLAB_API_TOKEN"
}

Require-Command -Name "curl"

if (-not $config.ContainsKey("LLM_PROVIDER") -or [string]::IsNullOrWhiteSpace($config["LLM_PROVIDER"])) {
    $config["LLM_PROVIDER"] = "vertex"
}
if (-not $config.ContainsKey("IMAGE_MODEL") -or [string]::IsNullOrWhiteSpace($config["IMAGE_MODEL"])) {
    $config["IMAGE_MODEL"] = "gemini-3.1-flash-image"
}
if (-not $config.ContainsKey("GITLAB_TRIGGER_REF") -or [string]::IsNullOrWhiteSpace($config["GITLAB_TRIGGER_REF"])) {
    $config["GITLAB_TRIGGER_REF"] = "main"
}
if (-not $config.ContainsKey("GITLAB_URL") -or [string]::IsNullOrWhiteSpace($config["GITLAB_URL"])) {
    $config["GITLAB_URL"] = "https://gitlab.com"
}
if (-not $config.ContainsKey("USE_ADK_ORCHESTRATOR") -or [string]::IsNullOrWhiteSpace($config["USE_ADK_ORCHESTRATOR"])) {
    $config["USE_ADK_ORCHESTRATOR"] = "true"
}
if (-not $config.ContainsKey("ADK_HARD_FAIL") -or [string]::IsNullOrWhiteSpace($config["ADK_HARD_FAIL"])) {
    $config["ADK_HARD_FAIL"] = "false"
}
if (-not $config.ContainsKey("PIPELINE_DRY_RUN") -or [string]::IsNullOrWhiteSpace($config["PIPELINE_DRY_RUN"])) {
    $config["PIPELINE_DRY_RUN"] = "false"
}

if ($DryRun) {
    Write-Host "[*] Dry-run mode enabled: validating config and planned actions only." -ForegroundColor Cyan
    Write-Host "[+] .env validation passed." -ForegroundColor Green
    Write-Host "[+] Modal secret update planned: $(-not $SkipSecret)" -ForegroundColor Green
    Write-Host "[+] Modal deploy planned: $(-not $SkipDeploy)" -ForegroundColor Green
    Write-Host "[+] GitLab API automation planned: $(-not $SkipGitLabApi)" -ForegroundColor Green
    if (-not $SkipGitLabApi) {
        Write-Host "[+] Sensitive vars protected mode: $($ProtectSensitiveVars.IsPresent)" -ForegroundColor Green
        Write-Host "[+] USE_ADK_ORCHESTRATOR=$($config["USE_ADK_ORCHESTRATOR"])" -ForegroundColor Green
        Write-Host "[+] ADK_HARD_FAIL=$($config["ADK_HARD_FAIL"])" -ForegroundColor Green
        Write-Host "[+] PIPELINE_DRY_RUN=$($config["PIPELINE_DRY_RUN"])" -ForegroundColor Green
    }
    if (-not [string]::IsNullOrWhiteSpace($WebhookUrl)) {
        Write-Host "[+] Webhook URL override provided via -WebhookUrl." -ForegroundColor Green
    }
    if ($config.ContainsKey("WEBHOOK_URL") -and -not [string]::IsNullOrWhiteSpace($config["WEBHOOK_URL"])) {
        Write-Host "[+] WEBHOOK_URL present in .env for webhook upsert fallback." -ForegroundColor Green
    }
    Write-Host "Dry-run complete. No remote changes were made." -ForegroundColor Yellow
    return
}

$env:MODAL_TOKEN_ID = $config["MODAL_TOKEN_ID"]
$env:MODAL_TOKEN_SECRET = $config["MODAL_TOKEN_SECRET"]

$modalPath = Get-ModalCommand
Write-Host "[*] Using Modal CLI: $modalPath" -ForegroundColor Cyan

$gitlabBaseUrl = Resolve-GitLabBaseUrl -Config $config
$gitlabHeaders = Get-GitLabApiHeaders -Config $config

$resolvedWebhookUrl = $WebhookUrl
if ([string]::IsNullOrWhiteSpace($resolvedWebhookUrl) -and $config.ContainsKey("WEBHOOK_URL") -and -not [string]::IsNullOrWhiteSpace($config["WEBHOOK_URL"])) {
    $resolvedWebhookUrl = $config["WEBHOOK_URL"]
}

if (-not $SkipSecret) {
    Write-Host "[*] Creating Modal secret gitmesh-keys..." -ForegroundColor Cyan
    
    $secretData = @{
        GCP_PROJECT_ID = $config["GCP_PROJECT_ID"]
        GCP_SERVICE_ACCOUNT_JSON = $config["GCP_SERVICE_ACCOUNT_JSON"]
        GITLAB_PROJECT_ID = $config["GITLAB_PROJECT_ID"]
        GITLAB_URL = $config["GITLAB_URL"]
        GITLAB_TRIGGER_TOKEN = $config["GITLAB_TRIGGER_TOKEN"]
        GITLAB_WEBHOOK_SECRET = $config["GITLAB_WEBHOOK_SECRET"]
        GITLAB_TRIGGER_REF = $config["GITLAB_TRIGGER_REF"]
        LLM_PROVIDER = $config["LLM_PROVIDER"]
        IMAGE_MODEL = $config["IMAGE_MODEL"]
    }
    $tempJsonPath = Join-Path $env:TEMP "gitmesh_secrets_temp.json"
    $jsonString = $secretData | ConvertTo-Json -Depth 100
    [System.IO.File]::WriteAllText($tempJsonPath, $jsonString)
    
    & $modalPath secret create gitmesh-keys --from-json $tempJsonPath --force
    Remove-Item $tempJsonPath -ErrorAction SilentlyContinue
}

if (-not $SkipDeploy) {
    Write-Host "[*] Deploying Modal webhook app..." -ForegroundColor Cyan
    $webhookDeployOutput = & $modalPath deploy ".\gitlab_webhook.py" 2>&1 | Out-String
    Write-Host $webhookDeployOutput

    if ([string]::IsNullOrWhiteSpace($resolvedWebhookUrl)) {
        $resolvedWebhookUrl = Get-UrlFromText -Text $webhookDeployOutput
    }

    Write-Host "[*] Deploying Modal compute app..." -ForegroundColor Cyan
    & $modalPath deploy ".\modal_app.py" | Out-Host
}

if (-not $SkipGitLabApi) {
    Write-Host "[*] Upserting GitLab CI/CD variables..." -ForegroundColor Cyan

    $protectSensitive = $ProtectSensitiveVars.IsPresent

    Set-GitLabVariable -BaseUrl $gitlabBaseUrl -ProjectId $config["GITLAB_PROJECT_ID"] -Headers $gitlabHeaders -Key "MODAL_TOKEN_ID" -Value $config["MODAL_TOKEN_ID"] -Masked $false -Protected $false
    Set-GitLabVariable -BaseUrl $gitlabBaseUrl -ProjectId $config["GITLAB_PROJECT_ID"] -Headers $gitlabHeaders -Key "MODAL_TOKEN_SECRET" -Value $config["MODAL_TOKEN_SECRET"] -Masked $true -Protected $protectSensitive
    Set-GitLabVariable -BaseUrl $gitlabBaseUrl -ProjectId $config["GITLAB_PROJECT_ID"] -Headers $gitlabHeaders -Key "GITLAB_API_TOKEN" -Value $config["GITLAB_API_TOKEN"] -Masked $true -Protected $protectSensitive
    Set-GitLabVariable -BaseUrl $gitlabBaseUrl -ProjectId $config["GITLAB_PROJECT_ID"] -Headers $gitlabHeaders -Key "GITLAB_TRIGGER_TOKEN" -Value $config["GITLAB_TRIGGER_TOKEN"] -Masked $true -Protected $protectSensitive
    Set-GitLabVariable -BaseUrl $gitlabBaseUrl -ProjectId $config["GITLAB_PROJECT_ID"] -Headers $gitlabHeaders -Key "GITLAB_WEBHOOK_SECRET" -Value $config["GITLAB_WEBHOOK_SECRET"] -Masked $true -Protected $protectSensitive
    Set-GitLabVariable -BaseUrl $gitlabBaseUrl -ProjectId $config["GITLAB_PROJECT_ID"] -Headers $gitlabHeaders -Key "GITLAB_TRIGGER_REF" -Value $config["GITLAB_TRIGGER_REF"] -Masked $false -Protected $false
    Set-GitLabVariable -BaseUrl $gitlabBaseUrl -ProjectId $config["GITLAB_PROJECT_ID"] -Headers $gitlabHeaders -Key "USE_ADK_ORCHESTRATOR" -Value $config["USE_ADK_ORCHESTRATOR"] -Masked $false -Protected $false
    Set-GitLabVariable -BaseUrl $gitlabBaseUrl -ProjectId $config["GITLAB_PROJECT_ID"] -Headers $gitlabHeaders -Key "ADK_HARD_FAIL" -Value $config["ADK_HARD_FAIL"] -Masked $false -Protected $false
    Set-GitLabVariable -BaseUrl $gitlabBaseUrl -ProjectId $config["GITLAB_PROJECT_ID"] -Headers $gitlabHeaders -Key "PIPELINE_DRY_RUN" -Value $config["PIPELINE_DRY_RUN"] -Masked $false -Protected $false

    if ([string]::IsNullOrWhiteSpace($resolvedWebhookUrl)) {
        throw "Could not determine webhook URL automatically. Re-run with -WebhookUrl https://<your-modal-webhook-url>."
    }

    Write-Host "[*] Upserting GitLab project webhook..." -ForegroundColor Cyan
    Upsert-GitLabWebhook -BaseUrl $gitlabBaseUrl -ProjectId $config["GITLAB_PROJECT_ID"] -Headers $gitlabHeaders -HookUrl $resolvedWebhookUrl -SecretToken $config["GITLAB_WEBHOOK_SECRET"]
}

Write-Host "" 
Write-Host "Remote setup bootstrap complete." -ForegroundColor Green
if (-not [string]::IsNullOrWhiteSpace($resolvedWebhookUrl)) {
    Write-Host "Webhook URL: $resolvedWebhookUrl" -ForegroundColor Yellow
}

if ($SkipGitLabApi) {
    Write-Host "GitLab API automation was skipped; configure GitLab variables and webhook manually." -ForegroundColor Yellow
} else {
    Write-Host "GitLab CI/CD variables and webhook have been configured/updated automatically." -ForegroundColor Yellow
}

Write-Host "" 
Write-Host "Ensure the Modal secret 'gitmesh-keys' contains:" -ForegroundColor Yellow
Write-Host "  GCP_PROJECT_ID=$($config["GCP_PROJECT_ID"])"
Write-Host "  GITLAB_PROJECT_ID=$($config["GITLAB_PROJECT_ID"])"
Write-Host "  GITLAB_URL=$($config["GITLAB_URL"])"
Write-Host "  GITLAB_TRIGGER_TOKEN=$($config["GITLAB_TRIGGER_TOKEN"])"
Write-Host "  GITLAB_WEBHOOK_SECRET=$($config["GITLAB_WEBHOOK_SECRET"])"
Write-Host "  GITLAB_TRIGGER_REF=$($config["GITLAB_TRIGGER_REF"])"
Write-Host "  LLM_PROVIDER=$($config["LLM_PROVIDER"])"
Write-Host "  IMAGE_MODEL=$($config["IMAGE_MODEL"])"
Write-Host "  GCP_SERVICE_ACCOUNT_JSON=<hidden>"
Write-Host "" 
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Open a GitLab issue titled 'MeshGen: <your asset prompt>'."
Write-Host "  2. Confirm a pipeline is triggered and issue comments stream stage status."