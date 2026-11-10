$ErrorActionPreference = "Stop"

$voicePath = Join-Path ([System.IO.Path]::GetTempPath()) ("ravan-ai-assistant-" + [guid]::NewGuid().ToString("N") + ".wav")
$voiceScript = Join-Path $PSScriptRoot "synthesize_showcase_voice.ps1"
$rendererScript = Join-Path $PSScriptRoot "render_showcase_video.py"

try {
    & $voiceScript -OutputPath $voicePath
    & python $rendererScript --voiceover $voicePath
    if ($LASTEXITCODE -ne 0) {
        throw "Could not render the simulation video."
    }
}
finally {
    if (Test-Path -LiteralPath $voicePath) {
        Remove-Item -LiteralPath $voicePath
    }
}
