param(
    [Parameter(Mandatory = $true)]
    [string]$Image,

    [string]$Namespace,

    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
$appDir = $PSScriptRoot
$manifestDir = Join-Path $appDir "openshift"
$deploymentName = "python-dash-shots"
$applicationPath = "/opt/openshift/applications/ai-lab/dash-fires"

if (-not (Test-Path (Join-Path $appDir "Dockerfile"))) {
    throw "Dockerfile was not found in the Dash_Shots_2026 deployment directory: $appDir"
}
foreach ($requiredManifest in @("deployment.yaml", "service.yaml", "route.yaml")) {
    if (-not (Test-Path (Join-Path $manifestDir $requiredManifest))) {
        throw "Required OpenShift manifest is missing: $requiredManifest"
    }
}

if (-not (Get-Command oc -ErrorAction SilentlyContinue)) {
    throw "The OpenShift CLI 'oc' was not found. Install it and log in with 'oc login' first."
}
oc whoami *> $null
if ($LASTEXITCODE -ne 0) {
    throw "You are not logged in to OpenShift. Run 'oc login' and try again."
}

if (-not $SkipBuild) {
    if (-not (Get-Command podman -ErrorAction SilentlyContinue)) {
        throw "Podman was not found. Install and log in to Podman, or pass -SkipBuild after pushing the image yourself."
    }
    podman build -t $Image $appDir
    if ($LASTEXITCODE -ne 0) { throw "Podman image build failed." }
    podman push $Image
    if ($LASTEXITCODE -ne 0) { throw "Podman image push failed." }
}

$ocArgs = @()
if ($Namespace) {
    $ocArgs += @("-n", $Namespace)
}

foreach ($manifest in @("deployment.yaml", "service.yaml", "route.yaml")) {
    oc @ocArgs apply -f (Join-Path $manifestDir $manifest)
    if ($LASTEXITCODE -ne 0) { throw "Applying $manifest failed." }
}

oc @ocArgs set image "deployment/$deploymentName" "app=$Image"
if ($LASTEXITCODE -ne 0) { throw "Setting the deployment image failed." }

oc @ocArgs rollout status "deployment/$deploymentName" --timeout=5m
if ($LASTEXITCODE -ne 0) { throw "Deployment rollout did not complete successfully." }

$route = oc @ocArgs get route $deploymentName -o jsonpath="{.spec.host}"
if ($LASTEXITCODE -ne 0) { throw "Could not retrieve the application route." }
Write-Host "Application deployed: https://$route"
Write-Host "Container application path: $applicationPath"
