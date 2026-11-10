param(
    [Parameter(Mandatory = $true)]
    [string]$OutputPath
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Speech
$synthesizer = [System.Speech.Synthesis.SpeechSynthesizer]::new()

try {
    $voices = @($synthesizer.GetInstalledVoices() | Where-Object { $_.Enabled })
    $indianEnglish = $voices | Where-Object { $_.VoiceInfo.Culture.Name -eq "en-IN" } | Select-Object -First 1
    $englishFallback = $voices | Where-Object { $_.VoiceInfo.Culture.Name -like "en-*" } | Select-Object -First 1
    $voice = if ($indianEnglish) { $indianEnglish } else { $englishFallback }
    if ($voice) {
        $synthesizer.SelectVoice($voice.VoiceInfo.Name)
        Write-Output "Showcase voice: $($voice.VoiceInfo.Name) ($($voice.VoiceInfo.Culture.Name))"
    }

    $synthesizer.Rate = 0
    $synthesizer.SetOutputToWaveFile($OutputPath)
    $synthesizer.Speak("Traffic is moving around us. Agent Drive adjusts speed and checks the gaps before changing lanes. The dashboard shows the following distance, collision risk, and lane safety. If traffic gets too close, we'll ease off and leave more room.")
}
finally {
    $synthesizer.Dispose()
}
