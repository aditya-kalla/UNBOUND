$zipPath = "c:\Aditya-Main\PROJECTS\UNBOUND\unbound-phase1\ai-spike-v2.zip"
if (Test-Path $zipPath) { Remove-Item $zipPath -Force }

$tempDir = "c:\Aditya-Main\PROJECTS\UNBOUND\unbound-phase1\ai-spike-v2-temp"
if (Test-Path $tempDir) { Remove-Item $tempDir -Recurse -Force }
New-Item -ItemType Directory -Path (Join-Path $tempDir "ai-spike") -Force | Out-Null

$baseDir = "c:\Aditya-Main\PROJECTS\UNBOUND\unbound-phase1\ai-spike"
$destBase = Join-Path $tempDir "ai-spike"

Copy-Item (Join-Path $baseDir "SPIKE_REPORT_V2.md") $destBase
Copy-Item (Join-Path $baseDir "ci-sample.yml") $destBase
Copy-Item (Join-Path $baseDir "licenses") $destBase -Recurse
Copy-Item (Join-Path $baseDir "logs") $destBase -Recurse
Copy-Item (Join-Path $baseDir "scripts") $destBase -Recurse

# Copy xray-rs without target
New-Item -ItemType Directory -Path (Join-Path $destBase "xray-rs") -Force | Out-Null
Copy-Item (Join-Path $baseDir "xray-rs\Cargo.toml") (Join-Path $destBase "xray-rs")
Copy-Item (Join-Path $baseDir "xray-rs\src") (Join-Path $destBase "xray-rs") -Recurse

# Copy autoplay-rs without target
New-Item -ItemType Directory -Path (Join-Path $destBase "autoplay-rs") -Force | Out-Null
Copy-Item (Join-Path $baseDir "autoplay-rs\Cargo.toml") (Join-Path $destBase "autoplay-rs")
Copy-Item (Join-Path $baseDir "autoplay-rs\src") (Join-Path $destBase "autoplay-rs") -Recurse
Copy-Item (Join-Path $baseDir "autoplay-rs\data") (Join-Path $destBase "autoplay-rs") -Recurse

# Copy data manifest only (no large wavs)
New-Item -ItemType Directory -Path (Join-Path $destBase "data\real") -Force | Out-Null
Copy-Item (Join-Path $baseDir "data\real\manifest.csv") (Join-Path $destBase "data\real")

Compress-Archive -Path $destBase -DestinationPath $zipPath
$info = Get-Item $zipPath
Write-Host "Created ai-spike-v2.zip successfully!"
Write-Host "File Path: $zipPath"
Write-Host "File Size: $($info.Length) bytes ($([math]::Round($info.Length / 1MB, 2)) MB)"

Remove-Item $tempDir -Recurse -Force
