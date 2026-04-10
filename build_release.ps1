param(
    [string]$Version = ""
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$pythonCommand = @("python")
if (Get-Command py -ErrorAction SilentlyContinue) {
    foreach ($version in @("-3.12", "-3.11")) {
        & py $version -c "import sys" *> $null
        if ($LASTEXITCODE -eq 0) {
            $pythonCommand = @("py", $version)
            break
        }
    }
}

if ($pythonCommand[0] -eq "py") {
    & py $pythonCommand[1] -m PyInstaller --noconfirm --clean MetaMan.spec
} else {
    & python -m PyInstaller --noconfirm --clean MetaMan.spec
}

if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller build failed."
}

$distFolder = Join-Path $root "dist\MetaMan"
if (-not (Test-Path -LiteralPath $distFolder)) {
    throw "Build output folder not found: $distFolder"
}

$archiveName = if ($Version) { "MetaMan-$Version-windows.zip" } else { "MetaMan-windows.zip" }
$zipPath = Join-Path $root "dist\$archiveName"

if (Test-Path -LiteralPath $zipPath) {
    Remove-Item -LiteralPath $zipPath -Force
}

for ($attempt = 1; $attempt -le 5; $attempt++) {
    try {
        Compress-Archive -Path (Join-Path $distFolder "*") -DestinationPath $zipPath
        break
    } catch {
        if ($attempt -eq 5) {
            throw
        }
        Start-Sleep -Seconds 2
    }
}

Write-Host "Built release artifact: $zipPath"
