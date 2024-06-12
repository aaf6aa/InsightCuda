# Insight
A CUDA C++ path tracer that leverages GPU acceleration for real-time realistic lighting rendering. The program solely utilizes CUDA for rendering and OpenGL for displaying buffers on screen.


![example](https://github.com/aaf6aa/InsightCuda/assets/56702415/4a69d4d8-0bcd-484c-8421-151f918ab8a2)

## Usage
The program can be run with the single executable provided in releases. The camera can be moved with WASD and rotated using the mouse while holding down middle mouse button. Resolution can be adjusted by resizing the window.

### Performance (960x720)
- GTX 1080ti: ~41 FPS
- RTX 3070ti: ~127 FPS

## Build Requirements
The program can be built with Visual Studio 2022 with the following dependecies:
- CUDA 12.x
- GLEW
- GLFW

Ensure that you have environmental variables `$(CUDA_PATH)`, `$(GLEW_PATH)`, and `$(GLFW_PATH)` which point to the respective libraries' directories.

## TODO
- Employ a better sampling strategy to reduce noise, i.e [ReSTIR](https://research.nvidia.com/sites/default/files/pubs/2020-07_Spatiotemporal-reservoir-resampling/ReSTIR.pdf) or a [world-space improvement of it](https://gpuopen.com/download/publications/SA2021_WorldSpace_ReSTIR.pdf).
- BVH for rendering complex scenes and .obj loading for models.
- Dynamic object adjustment through the GUI.

## Acknowledgments
- [Crash Course in BRDF Implementation](https://boksajak.github.io/blog/BRDF)
- [Dear ImGui](https://github.com/ocornut/imgui)
- [stb single-file public domain libraries for C/C++](https://github.com/nothings/stb)
