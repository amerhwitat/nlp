$ErrorActionPreference = 'Stop'
$root = Resolve-Path (Join-Path $PSScriptRoot '../..')
$build = Join-Path $root 'build/gpu'
cmake -S (Join-Path $root 'gpu') -B $build -DCMAKE_BUILD_TYPE=Release -DAVRS_ENABLE_OPENGL=ON -DAVRS_ENABLE_DIRECTX12=ON
cmake --build $build --config Release --parallel
Write-Host 'AVRS CPU/GPU bridge build completed.'
Write-Host 'CUDA kernels require a CUDA-enabled CMake toolchain.'
Write-Host 'Unreal and Unity adapters are built by their respective editor/toolchains.'
