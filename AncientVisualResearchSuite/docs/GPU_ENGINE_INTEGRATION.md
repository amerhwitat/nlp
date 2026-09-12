# GPU and engine integration

AVRS uses a backend-neutral compute contract and optional adapters.

## Graphics/compute backends

- OpenGL 4.6 compute shaders: portable desktop/reference GPU path.
- DirectX 12 / Shader Model 6: Windows-native path and DirectML interoperability.
- CUDA: NVIDIA tensor/image kernels.
- OpenCV: accelerated image preprocessing when the installed OpenCV build exposes CUDA.
- Unreal Engine 5: runtime plugin and historical-time World Subsystem.
- Unity/Unity3D: C# scene bridge and package integration.

## Neural execution

The RNN/LLM layer uses ONNX Runtime where possible. Provider selection is runtime-driven, allowing CUDA, TensorRT, DirectML, OpenVINO, CoreML, NNAPI, WebGPU and CPU fallback. The engine registry records provenance and does not imply that every provider is available on every machine.

## Build boundary

The repository contains source and build configuration for each adapter. Native GPU binaries are hardware/toolchain dependent. UE5 plugins must be compiled by the Unreal Build Tool; Unity scripts/packages by Unity; CUDA kernels by a CUDA-enabled compiler; iOS binaries by Xcode.

## Research visualization

GPU buffers are designed for 128D entity tensors, event graphs, character transforms, terrain/sky samples, glyph features and animation frames. CPU fallback is mandatory so historical datasets remain inspectable without a GPU.

## Evidence safety

GPU acceleration changes execution speed, not historical truth. Every generated reconstruction retains evidence class, source references, uncertainty and model provenance.
