param(
    [string]$Version = ""
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$pythonCommand = @()
$activeVenvPython = if ($env:VIRTUAL_ENV) { Join-Path $env:VIRTUAL_ENV "Scripts\python.exe" } else { "" }
$localVenvPython = Join-Path $root ".venv\Scripts\python.exe"

if ($activeVenvPython -and (Test-Path -LiteralPath $activeVenvPython)) {
    $pythonCommand = @($activeVenvPython)
} elseif (Test-Path -LiteralPath $localVenvPython) {
    $pythonCommand = @($localVenvPython)
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    foreach ($version in @("-3.12", "-3.11")) {
        & py $version -c "import sys" *> $null
        if ($LASTEXITCODE -eq 0) {
            $pythonCommand = @("py", $version)
            break
        }
    }
} else {
    $pythonCommand = @("python")
}

if (-not $pythonCommand) {
    $pythonCommand = @("python")
}

Write-Host "Using Python: $($pythonCommand -join ' ')"

$pythonExe = $pythonCommand[0]
$venvRoot = Split-Path -Parent (Split-Path -Parent $pythonExe)
$pyvenvCfg = Join-Path $venvRoot "pyvenv.cfg"
if (Test-Path -LiteralPath $pyvenvCfg) {
    $homeLine = Get-Content -LiteralPath $pyvenvCfg | Where-Object { $_ -match "^\s*home\s*=" } | Select-Object -First 1
    if ($homeLine) {
        $pythonHome = ($homeLine -split "=", 2)[1].Trim()
        $condaLibraryBin = Join-Path $pythonHome "Library\bin"
        if (Test-Path -LiteralPath $condaLibraryBin) {
            $env:PATH = "$condaLibraryBin;$env:PATH"
            Write-Host "Added Conda DLL path: $condaLibraryBin"
        }
    }
}

if ($pythonCommand[0] -eq "py") {
    & py $pythonCommand[1] -m PyInstaller --noconfirm --clean MetaMan.spec
} else {
    & $pythonCommand[0] -m PyInstaller --noconfirm --clean MetaMan.spec
}

if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller build failed."
}

$distFolder = Join-Path $root "dist\MetaMan"
if (-not (Test-Path -LiteralPath $distFolder)) {
    throw "Build output folder not found: $distFolder"
}

$iconSource = Join-Path $root "MetaMan\assets\metaman.ico"
if (Test-Path -LiteralPath $iconSource) {
    Copy-Item -LiteralPath $iconSource -Destination (Join-Path $distFolder "MetaMan.ico") -Force
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
