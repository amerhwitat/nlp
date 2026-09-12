@echo off
setlocal
cd /d "%~dp0.."
where cmake >nul 2>nul && (cmake -S cpp -B cpp\build && cmake --build cpp\build) || echo SKIP: cmake
where dotnet >nul 2>nul && dotnet build csharp\ThamudicOcr.csproj || echo SKIP: dotnet
where mvn >nul 2>nul && mvn -q -f java\pom.xml package || echo SKIP: maven
where go >nul 2>nul && go build -o go\ocr_scan.exe .\go || echo SKIP: go
where cargo >nul 2>nul && cargo build --manifest-path rust\Cargo.toml || echo SKIP: cargo
where node >nul 2>nul && node --check javascript\ocr_scan.mjs || echo SKIP: node
where npm >nul 2>nul && (cd typescript && call npm install && call npm run build) || echo SKIP: npm
endlocal
