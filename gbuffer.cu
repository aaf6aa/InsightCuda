#pragma once

#include <float.h>
#include <math.h>
#include <stdint.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>
#include <math_constants.h>

#include "util.cu"
#include "math.cu"
#include "primitives.cu"

struct GBuffer
{
    enum {
        DENOISED = 0,
        RADIANCE = 1,
        POSITION = 2,
        DEPTH = 3,
        NORMAL = 4,
        ALBEDO = 5,
        EMISSION = 6,
        ROUGHNESS = 7,
        METALNESS = 8
    };

    float3* denoised = nullptr;
    float3* radiance = nullptr;
    uint32_t* seeds = nullptr;

    float3* position = nullptr;
    float* depth = nullptr;
    float3* normal = nullptr;
    float3* albedo = nullptr;
    float3* emission = nullptr;
    float* roughness = nullptr;
    float* metalness = nullptr;

    float3* position_min = nullptr;
    float3* position_max = nullptr;
    float3* depth_min = nullptr;
    float3* depth_max = nullptr;
    float3* normal_min = nullptr;
    float3* normal_max = nullptr;

    float3* position_norm = nullptr;
    float3* depth_norm = nullptr;
    float3* normal_norm = nullptr;

    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t samples = 0;
    uint32_t frame_idx = 0;

    GLuint fbo = 0;
    GLuint gl_textures[9];
    cudaGraphicsResource* cuda_textures[9];
    cudaArray_t cuda_arrays[9];
    cudaSurfaceObject_t surfaces[9];

    void _init_tex(GLuint& gl_tex, cudaGraphicsResource*& cuda_tex, bool framebuffer = false)
    {
        // Create a texture to render to
        glGenTextures(1, &gl_tex);
        glBindTexture(GL_TEXTURE_2D, gl_tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, (GLsizei)width, (GLsizei)height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        if (framebuffer)
        {
            // Attach texture to framebuffer
            glGenFramebuffers(1, &fbo);
            glBindFramebuffer(GL_FRAMEBUFFER, fbo);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, gl_tex, 0);
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
        }

        // Register the texture with CUDA
        CUDA_ERROR_CHECK(cudaGraphicsGLRegisterImage(&cuda_tex, gl_tex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
    }

    void _create_surface(cudaSurfaceObject_t& surface, cudaGraphicsResource*& cuda_tex, cudaArray_t& cuda_array, cudaStream_t& stream)
    {
        cudaGraphicsMapResources(1, &cuda_tex, stream);
        cudaGraphicsSubResourceGetMappedArray(&cuda_array, cuda_tex, 0, 0);

        // Bind the CUDA array to a surface
        cudaResourceDesc resDesc = {};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cuda_array;
        CUDA_ERROR_CHECK(cudaCreateSurfaceObject(&surface, &resDesc));
    }

    void init(uint32_t height, uint32_t width)
    {
        this->width = width;
        this->height = height;
        this->samples = 0;
        this->fbo = 0;

        CUDA_ERROR_CHECK(cudaMalloc(&denoised, width * height * sizeof(float3)));
        CUDA_ERROR_CHECK(cudaMalloc(&radiance, width * height * sizeof(float3)));
        CUDA_ERROR_CHECK(cudaMalloc(&seeds, width * height * sizeof(uint32_t)));

        CUDA_ERROR_CHECK(cudaMalloc(&position, width * height * sizeof(float3)));
        CUDA_ERROR_CHECK(cudaMalloc(&depth, width * height * sizeof(float)));
        CUDA_ERROR_CHECK(cudaMalloc(&normal, width * height * sizeof(float3)));
        CUDA_ERROR_CHECK(cudaMalloc(&albedo, width * height * sizeof(float3)));
        CUDA_ERROR_CHECK(cudaMalloc(&emission, width * height * sizeof(float3)));
        CUDA_ERROR_CHECK(cudaMalloc(&roughness, width * height * sizeof(float)));
        CUDA_ERROR_CHECK(cudaMalloc(&metalness, width * height * sizeof(float)));

        CUDA_ERROR_CHECK(cudaMalloc(&position_min, ((height * width + 256 - 1) / 256) * sizeof(float3)));
        CUDA_ERROR_CHECK(cudaMalloc(&position_max, ((height * width + 256 - 1) / 256) * sizeof(float3)));
        CUDA_ERROR_CHECK(cudaMalloc(&depth_min, ((height * width + 256 - 1) / 256) * sizeof(float3)));
        CUDA_ERROR_CHECK(cudaMalloc(&depth_max, ((height * width + 256 - 1) / 256) * sizeof(float3)));
        CUDA_ERROR_CHECK(cudaMalloc(&normal_min, ((height * width + 256 - 1) / 256) * sizeof(float3)));
        CUDA_ERROR_CHECK(cudaMalloc(&normal_max, ((height * width + 256 - 1) / 256) * sizeof(float3)));

        CUDA_ERROR_CHECK(cudaMalloc(&position_norm, width * height * sizeof(float3)));
        CUDA_ERROR_CHECK(cudaMalloc(&depth_norm, width * height * sizeof(float3)));
        CUDA_ERROR_CHECK(cudaMalloc(&normal_norm, width * height * sizeof(float3)));

        _init_tex(gl_textures[DENOISED], cuda_textures[DENOISED], true);
        _init_tex(gl_textures[RADIANCE], cuda_textures[RADIANCE]);
        _init_tex(gl_textures[POSITION], cuda_textures[POSITION]);
        _init_tex(gl_textures[DEPTH], cuda_textures[DEPTH]);
        _init_tex(gl_textures[NORMAL], cuda_textures[NORMAL]);
        _init_tex(gl_textures[ALBEDO], cuda_textures[ALBEDO]);
        _init_tex(gl_textures[EMISSION], cuda_textures[EMISSION]);
        _init_tex(gl_textures[ROUGHNESS], cuda_textures[ROUGHNESS]);
        _init_tex(gl_textures[METALNESS], cuda_textures[METALNESS]);
    }

    void create_surfaces(cudaStream_t& stream)
    {
        _create_surface(surfaces[DENOISED], cuda_textures[DENOISED], cuda_arrays[DENOISED], stream);
        _create_surface(surfaces[RADIANCE], cuda_textures[RADIANCE], cuda_arrays[RADIANCE], stream);
        _create_surface(surfaces[POSITION], cuda_textures[POSITION], cuda_arrays[POSITION], stream);
        _create_surface(surfaces[DEPTH], cuda_textures[DEPTH], cuda_arrays[DEPTH], stream);
        _create_surface(surfaces[NORMAL], cuda_textures[NORMAL], cuda_arrays[NORMAL], stream);
        _create_surface(surfaces[ALBEDO], cuda_textures[ALBEDO], cuda_arrays[ALBEDO], stream);
        _create_surface(surfaces[EMISSION], cuda_textures[EMISSION], cuda_arrays[EMISSION], stream);
        _create_surface(surfaces[ROUGHNESS], cuda_textures[ROUGHNESS], cuda_arrays[ROUGHNESS], stream);
        _create_surface(surfaces[METALNESS], cuda_textures[METALNESS], cuda_arrays[METALNESS], stream);
    }

    void blit()
    {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        glBlitFramebuffer(0, 0, (GLint)width, (GLint)height, 0, (GLint)height, (GLint)width, 0, GL_COLOR_BUFFER_BIT, GL_LINEAR);
    }

    void destroy_surfaces(cudaStream_t& stream)
    {
        CUDA_ERROR_CHECK(cudaDestroySurfaceObject(surfaces[DENOISED]));
        CUDA_ERROR_CHECK(cudaDestroySurfaceObject(surfaces[RADIANCE]));
        CUDA_ERROR_CHECK(cudaDestroySurfaceObject(surfaces[POSITION]));
        CUDA_ERROR_CHECK(cudaDestroySurfaceObject(surfaces[DEPTH]));
        CUDA_ERROR_CHECK(cudaDestroySurfaceObject(surfaces[NORMAL]));
        CUDA_ERROR_CHECK(cudaDestroySurfaceObject(surfaces[ALBEDO]));
        CUDA_ERROR_CHECK(cudaDestroySurfaceObject(surfaces[EMISSION]));
        CUDA_ERROR_CHECK(cudaDestroySurfaceObject(surfaces[ROUGHNESS]));
        CUDA_ERROR_CHECK(cudaDestroySurfaceObject(surfaces[METALNESS]));
        CUDA_ERROR_CHECK(cudaGraphicsUnmapResources(9, cuda_textures, stream));
    }

    void free()
    {
        samples = 0;

        CUDA_ERROR_CHECK(cudaFree(denoised));
        CUDA_ERROR_CHECK(cudaFree(radiance));
        CUDA_ERROR_CHECK(cudaFree(seeds));

        CUDA_ERROR_CHECK(cudaFree(position));
        CUDA_ERROR_CHECK(cudaFree(depth));
        CUDA_ERROR_CHECK(cudaFree(normal));
        CUDA_ERROR_CHECK(cudaFree(albedo));
        CUDA_ERROR_CHECK(cudaFree(emission));
        CUDA_ERROR_CHECK(cudaFree(roughness));
        CUDA_ERROR_CHECK(cudaFree(metalness));

        CUDA_ERROR_CHECK(cudaFree(position_min));
        CUDA_ERROR_CHECK(cudaFree(position_max));
        CUDA_ERROR_CHECK(cudaFree(depth_min));
        CUDA_ERROR_CHECK(cudaFree(depth_max));
        CUDA_ERROR_CHECK(cudaFree(normal_min));
        CUDA_ERROR_CHECK(cudaFree(normal_max));

        glDeleteTextures(9, gl_textures);
        glDeleteFramebuffers(1, &fbo);
    }
};

__global__ void set_seeds(GBuffer gbuffer)
{
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t idx = y * gbuffer.width + x;

    if (x >= gbuffer.width || y >= gbuffer.height)
        return;

    gbuffer.seeds[idx] = idx + 1;
}
