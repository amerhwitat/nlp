$ErrorActionPreference='Stop'
$Root=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$android=Join-Path $Root 'mobile/android'
if (Test-Path (Join-Path $android 'gradlew.bat')) { Push-Location $android; .\gradlew.bat assembleRelease bundleRelease; Pop-Location }
$flutter=Join-Path $Root 'mobile/flutter'
if ((Test-Path $flutter) -and (Get-Command flutter -ErrorAction SilentlyContinue)) { Push-Location $flutter; flutter pub get; flutter build apk --release; Pop-Location }
Write-Host 'iOS IPA requires macOS/Xcode signing. Run scripts/build/build-mobile.sh on macOS or xcodebuild with the configured signing identity.'
