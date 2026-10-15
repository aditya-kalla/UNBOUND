$zipPath = 'c:\Aditya-Main\PROJECTS\UNBOUND\unbound-phase1\ai-spike.zip'
if (Test-Path $zipPath) { Remove-Item $zipPath -Force }

$tempStage = 'c:\Aditya-Main\PROJECTS\UNBOUND\unbound-phase1\ai-spike-stage'
if (Test-Path $tempStage) { Remove-Item $tempStage -Recurse -Force }
New-Item -ItemType Directory -Path $tempStage | Out-Null

$baseDir = (Get-Item 'ai-spike').FullName

Get-ChildItem -Path 'ai-spike' -Recurse | Where-Object {
    $_.FullName -notmatch '\\\.venv($|\\)' -and
    $_.FullName -notmatch '\\target($|\\)' -and
    $_.FullName -notmatch '__pycache__' -and
    -not ($_.Length -gt 52428800)
} | ForEach-Object {
    $rel = $_.FullName.Substring($baseDir.Length + 1)
    $dest = Join-Path $tempStage $rel
    if ($_.PSIsContainer) {
        if (-not (Test-Path $dest)) {
            New-Item -ItemType Directory -Path $dest -Force | Out-Null
        }
    } else {
        $parent = Split-Path $dest
        if (-not (Test-Path $parent)) {
            New-Item -ItemType Directory -Path $parent -Force | Out-Null
        }
        Copy-Item -Path $_.FullName -Destination $dest -Force
    }
}

Compress-Archive -Path "$tempStage\*" -DestinationPath $zipPath -Force
Remove-Item $tempStage -Recurse -Force

Write-Host "ZIP created successfully at: $zipPath"
Get-Item $zipPath | Select-Object FullName, Length
