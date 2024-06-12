#ifndef __CUDACC__  
#define __CUDACC__
#endif

#pragma comment(lib, "opengl32.lib") 
#pragma comment (lib, "glew32s.lib")
#define GLEW_STATIC

#include <algorithm>
#include <chrono>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string>
#include <vector>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>
#include <math_constants.h>

#include "util.cu"
#include "math.cu"
#include "primitives.cu"
#include "camera.cu"
#include "scene.cu"
#include "gbuffer.cu"
#include "brdf.cu"

// STB image libraries
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"


__device__ void sample_triangle(float3& point, float3& normal, float& pdf, const Triangle& tri, uint32_t& seed)
{
    float u = xorshift32sf(seed);
	float v = xorshift32sf(seed);

	if (u + v > 1.0f)
	{
		u = 1.0f - u;
		v = 1.0f - v;
	}

	float w = 1.0f - u - v;

    float3 a = cross(tri.vertices[1] - tri.vertices[0], tri.vertices[2] - tri.vertices[0]);
    float area = length(a) * 0.5f;

    point = tri.vertices[0] * u + tri.vertices[1] * v + tri.vertices[2] * w;
    normal = normalize(a);
    pdf = 1.0f / area;
}

__device__ float power_heuristic(float a_pdf, uint32_t num_a, float b_pdf, uint32_t num_b)
{
    float a = a_pdf * num_a;
    float b = b_pdf * num_b;
	return (a * a) / (a * a + b * b);
}

__global__ void pathtrace(GBuffer gbuffer, Scene scene)
{
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t idx = y * gbuffer.width + x;

    uint32_t& seed = gbuffer.seeds[idx];
    seed += gbuffer.frame_idx;

    if (x >= gbuffer.width || y >= gbuffer.height)
    {
        return;
    }

    if (gbuffer.samples == 0)
    {
        gbuffer.denoised[idx] = float3_zero;
        gbuffer.radiance[idx] = float3_zero;
        gbuffer.position[idx] = float3_zero;
        gbuffer.depth[idx] = 0.0f;
        gbuffer.depth_norm[idx] = float3_zero; // need this to keep a float3 copy for normalization
        gbuffer.normal[idx] = float3_zero;
        gbuffer.albedo[idx] = float3_zero;
        gbuffer.emission[idx] = float3_zero;
        gbuffer.roughness[idx] = 0.0f;
        gbuffer.metalness[idx] = 0.0f;
    }

    float3 radiance = float3_zero;
    float3 throughput = float3_one;

    Ray ray = scene.camera.generate_ray((float)x / gbuffer.width, (float)y / gbuffer.height, seed);

    bool second_bounce_gbuffer = false;
    uint32_t min_bounces = 4;
    uint32_t max_bounces = 8;
    for (uint32_t bounce_i = 0; bounce_i < max_bounces; bounce_i++)
    {
        HitRecord hit;
        if (!scene.intersect(hit, ray))
        {
            const float3 sky_color = make_float3(0.5f, 0.5f, 0.5f);
            if (bounce_i == 0)
            {
                gbuffer.position[idx] = (gbuffer.position[idx] * gbuffer.samples + (float3_zero)) / (gbuffer.samples + 1);
                gbuffer.depth[idx] = (gbuffer.depth[idx] * gbuffer.samples + (0.0f)) / (gbuffer.samples + 1);
                gbuffer.depth_norm[idx] = (gbuffer.depth_norm[idx] * gbuffer.samples + (float3_zero)) / (gbuffer.samples + 1);
                gbuffer.normal[idx] = (gbuffer.normal[idx] * gbuffer.samples + (float3_zero)) / (gbuffer.samples + 1);
                gbuffer.albedo[idx] = (gbuffer.albedo[idx] * gbuffer.samples + (sky_color)) / (gbuffer.samples + 1);
                gbuffer.emission[idx] = (gbuffer.emission[idx] * gbuffer.samples + (float3_zero)) / (gbuffer.samples + 1);
                gbuffer.roughness[idx] = (gbuffer.roughness[idx] * gbuffer.samples + (0.0f)) / (gbuffer.samples + 1);
                gbuffer.metalness[idx] = (gbuffer.metalness[idx] * gbuffer.samples + (0.0f)) / (gbuffer.samples + 1);
            }
            radiance += throughput * sky_color;
            break;
        }

        float3 wo = -ray.direction;
        Material& material = scene.materials[hit.material_id];

        if (bounce_i == 0)
        {
            if (!second_bounce_gbuffer && (material.roughness < 0.2f && material.metalness > 0.8f))
            {
                second_bounce_gbuffer = true;
            }
            if (!second_bounce_gbuffer)
			{
				gbuffer.position[idx] = (gbuffer.position[idx] * gbuffer.samples + (hit.position)) / (gbuffer.samples + 1);
				gbuffer.depth[idx] = (gbuffer.depth[idx] * gbuffer.samples + (hit.distance)) / (gbuffer.samples + 1);
				gbuffer.depth_norm[idx] = (gbuffer.depth_norm[idx] * gbuffer.samples + (hit.distance)) / (gbuffer.samples + 1);
				gbuffer.normal[idx] = (gbuffer.normal[idx] * gbuffer.samples + (hit.normal)) / (gbuffer.samples + 1);
				gbuffer.albedo[idx] = (gbuffer.albedo[idx] * gbuffer.samples + (material.albedo)) / (gbuffer.samples + 1);
			}
            gbuffer.emission[idx] = (gbuffer.emission[idx] * gbuffer.samples + (material.emission)) / (gbuffer.samples + 1);
            gbuffer.roughness[idx] = (gbuffer.roughness[idx] * gbuffer.samples + (material.roughness)) / (gbuffer.samples + 1);
            gbuffer.metalness[idx] = (gbuffer.metalness[idx] * gbuffer.samples + (material.metalness)) / (gbuffer.samples + 1);
        }

        if (bounce_i > 0 && second_bounce_gbuffer)
        {
            gbuffer.position[idx] = (gbuffer.position[idx] * gbuffer.samples + (hit.position)) / (gbuffer.samples + 1);
            gbuffer.depth[idx] = (gbuffer.depth[idx] * gbuffer.samples + (hit.distance)) / (gbuffer.samples + 1);
            gbuffer.depth_norm[idx] = (gbuffer.depth_norm[idx] * gbuffer.samples + (hit.distance)) / (gbuffer.samples + 1);
            gbuffer.normal[idx] = (gbuffer.normal[idx] * gbuffer.samples + (hit.normal)) / (gbuffer.samples + 1);
            gbuffer.albedo[idx] = (gbuffer.albedo[idx] * gbuffer.samples + (material.albedo)) / (gbuffer.samples + 1);
            second_bounce_gbuffer = false;
        }

        if (length(material.emission) > 0.0f)
        {
            radiance += throughput * material.emission;
			break;
        }

        /*
        const uint32_t candidate_light_count = 4;
        uint32_t selected_light_idx = 0;
        float total_weight = 0.0f;

        for (uint32_t i = 0; i < min(candidate_light_count, scene.light_count); i++)
		{
            uint32_t light_idx = i;
            if (scene.light_count > candidate_light_count)
			{
				light_idx = xorshift32s(seed) % scene.light_count;
			}

            light_idx = scene.lights[light_idx];

			Triangle& light = scene.tris[light_idx];

            float3 light_point;
            float3 light_normal;
            float light_pdf;
            sample_triangle(light_point, light_normal, light_pdf, light, seed);

            float3 light_dir = normalize(light_point - hit.position);
            if (dot(hit.normal, light_dir) < EPS_F)
            {
                continue;
            }

            // convert pdf to solid angle
            light_pdf *= length_squared(light_point - hit.position) / fabsf(dot(light_normal, -light_dir));
            
            float3 light_weight = scene.materials[light.material_id].emission / light_pdf;

            total_weight += length(light_weight);
            if (xorshift32sf(seed) < (length(light_weight) / total_weight))
            {
				selected_light_idx = light_idx;
            }

		}

        if (total_weight > 0.0f)
        {
            Triangle& light = scene.tris[selected_light_idx];
            float3 light_point;
            float3 light_normal;
            float light_pdf;
            sample_triangle(light_point, light_normal, light_pdf, light, seed);

            float3 light_dir = normalize(light_point - hit.position);

            HitRecord shadow_hit;
            Ray shadow_ray = { hit.position + hit.normal * EPS_F, light_dir };

            if (trace_ray(shadow_hit, shadow_ray, scene.tris, scene.tri_count) && shadow_hit.triangle_id == selected_light_idx)
            {
                light_pdf *= length_squared(light_point - hit.position) / fabsf(dot(light_normal, -light_dir));
                float3 light_emission = scene.materials[light.material_id].emission;
                
                float brdf_pdf;
                float3 brdf_weight = brdf::evalCombinedBRDF(brdf_pdf, hit.normal, light_dir, wo, (brdf::MaterialProperties&)material);
                
                float weight = power_heuristic(light_pdf, 1, brdf_pdf, 1);
                radiance += throughput * brdf_weight * light_emission * weight / light_pdf;
            }
        }
        */

        if (bounce_i == max_bounces)
        {
            // break early since the rest of the code won't contribute
            break;
        }

        if (bounce_i > min_bounces)
        {
            // russian roulette
            float p = length(throughput);
            if (xorshift32sf(seed) > p)
            {
                break;
            }
            throughput /= p;
        }

        float3 wi;
        float3 brdf_weight;

        bool bounce_found = false;
        float specular_p = 0.5f;
        bool is_specular = false;
        // sample brdf repeatedly to find a valid bounce direction
        for (uint32_t i = 0; i < 8; i++)
        {
            specular_p = brdf::getBrdfProbability(material.albedo, material.metalness, wo, hit.normal);
            is_specular = xorshift32sf(seed) < specular_p;

            float2 rand = make_float2(xorshift32sf(seed), xorshift32sf(seed));
            bounce_found = brdf::evalIndirectCombinedBRDF(wi, brdf_weight, hit.normal, hit.normal, wo, (brdf::MaterialProperties&)material, rand, is_specular);
            if (bounce_found)
            {
                break;
            }
        }

        if (!bounce_found)
		{
			break;
		}

        if (is_specular)
        {
            throughput /= specular_p;
        }
        else
        {
            throughput /= (1.0f - specular_p);
        }

        throughput *= brdf_weight;

        ray = Ray(hit.position + hit.normal * EPS_F, wi);
    }

    gbuffer.radiance[idx] = (gbuffer.radiance[idx] * gbuffer.samples + (radiance)) / (gbuffer.samples + 1);
    gbuffer.denoised[idx] = gbuffer.radiance[idx];
}

__global__ void denoise(GBuffer gbuffer, float sigma = 0.15f)
{
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t idx = y * gbuffer.width + x;

	if (x >= gbuffer.width || y >= gbuffer.height)
	{
		return;
	}

    float3 rgb = gbuffer.radiance[idx];
    float depth = gbuffer.depth[idx];
    float3 normal = gbuffer.normal[idx];

    float3 sum = float3_zero;
    float weight_sum = 0.0f;

    const int kernel_radius = 5;
#pragma unroll
    for (int dy = -kernel_radius; dy <= kernel_radius; dy++)
    {
        uint32_t ny = min(max(y + dy, 0), gbuffer.height);
#pragma unroll
        for (int dx = -kernel_radius; dx <= kernel_radius; dx++)
		{
            uint32_t nx = min(max(x + dx, 0), gbuffer.width);
			uint32_t n_idx = ny * gbuffer.width + nx;

            float3 c = gbuffer.radiance[n_idx];
			float3 n = gbuffer.normal[n_idx];
			float d = gbuffer.depth[n_idx];

            float depth_diff = depth - d;

            float color_weight = __expf(-length_squared(rgb - c) / (2.0f * sigma * sigma));
            float normal_weight = __expf(-length_squared(normal - n) / (2.0f * sigma * sigma));
            float depth_weight = __expf(-(depth_diff * depth_diff) / (2.0f * sigma * sigma));

            float weight = color_weight * normal_weight * depth_weight;
            sum += c * weight;
            weight_sum += weight;
		}
    }

    gbuffer.denoised[idx] = sum / weight_sum;
}

// NVIDIA: Optimizing Parallel Reduction in CUDA
// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
__device__ void reduce_minmax_warp(volatile float3* smin, volatile float3* smax, int tid)
{
    smin[tid].x = fminf(smin[tid].x, smin[tid + 32].x);
    smin[tid].y = fminf(smin[tid].y, smin[tid + 32].y);
    smin[tid].z = fminf(smin[tid].z, smin[tid + 32].z);
    smax[tid].x = fmaxf(smax[tid].x, smax[tid + 32].x);
    smax[tid].y = fmaxf(smax[tid].y, smax[tid + 32].y);
    smax[tid].z = fmaxf(smax[tid].z, smax[tid + 32].z);

    smin[tid].x = fminf(smin[tid].x, smin[tid + 16].x);
    smin[tid].y = fminf(smin[tid].y, smin[tid + 16].y);
    smin[tid].z = fminf(smin[tid].z, smin[tid + 16].z);
    smax[tid].x = fmaxf(smax[tid].x, smax[tid + 16].x);
    smax[tid].y = fmaxf(smax[tid].y, smax[tid + 16].y);
    smax[tid].z = fmaxf(smax[tid].z, smax[tid + 16].z);

    smin[tid].x = fminf(smin[tid].x, smin[tid + 8].x);
    smin[tid].y = fminf(smin[tid].y, smin[tid + 8].y);
    smin[tid].z = fminf(smin[tid].z, smin[tid + 8].z);
    smax[tid].x = fmaxf(smax[tid].x, smax[tid + 8].x);
    smax[tid].y = fmaxf(smax[tid].y, smax[tid + 8].y);
    smax[tid].z = fmaxf(smax[tid].z, smax[tid + 8].z);

    smin[tid].x = fminf(smin[tid].x, smin[tid + 4].x);
    smin[tid].y = fminf(smin[tid].y, smin[tid + 4].y);
    smin[tid].z = fminf(smin[tid].z, smin[tid + 4].z);
    smax[tid].x = fmaxf(smax[tid].x, smax[tid + 4].x);
    smax[tid].y = fmaxf(smax[tid].y, smax[tid + 4].y);
    smax[tid].z = fmaxf(smax[tid].z, smax[tid + 4].z);

    smin[tid].x = fminf(smin[tid].x, smin[tid + 2].x);
    smin[tid].y = fminf(smin[tid].y, smin[tid + 2].y);
    smin[tid].z = fminf(smin[tid].z, smin[tid + 2].z);
    smax[tid].x = fmaxf(smax[tid].x, smax[tid + 2].x);
    smax[tid].y = fmaxf(smax[tid].y, smax[tid + 2].y);
    smax[tid].z = fmaxf(smax[tid].z, smax[tid + 2].z);

    smin[tid].x = fminf(smin[tid].x, smin[tid + 1].x);
    smin[tid].y = fminf(smin[tid].y, smin[tid + 1].y);
    smin[tid].z = fminf(smin[tid].z, smin[tid + 1].z);
    smax[tid].x = fmaxf(smax[tid].x, smax[tid + 1].x);
    smax[tid].y = fmaxf(smax[tid].y, smax[tid + 1].y);
    smax[tid].z = fmaxf(smax[tid].z, smax[tid + 1].z);
}

__global__ void reduce_minmax(float3* min_out, float3* max_out, float3* min_in, float3* max_in, uint32_t size)
{
    extern __shared__ float3 sdata[];
    float3* smin = sdata;
    float3* smax = (float3*)&sdata[blockDim.x];

    uint32_t tid = threadIdx.x;
    uint32_t i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    if (i < size)
    {
        smin[tid] = fminf(min_in[i], min_in[i + blockDim.x]);
        smax[tid] = fmaxf(max_in[i], max_in[i + blockDim.x]);
    }
    else
    {
        smin[tid] = make_float3(FLT_MAX);
        smax[tid] = make_float3(FLT_MIN);
    }

    __syncthreads();
    for (uint32_t s = blockDim.x / 2; s > 32; s >>= 1)
    {
        if (tid < s)
        {
            smin[tid] = fminf(smin[tid], smin[tid + s]);
            smax[tid] = fmaxf(smax[tid], smax[tid + s]);
        }
        __syncthreads();
    }

    if (tid < 32)
    {
        reduce_minmax_warp(smin, smax, tid);
    }

    if (tid == 0)
    {
        min_out[blockIdx.x] = smin[0];
		max_out[blockIdx.x] = smax[0];
    }
}

__global__ void normalize(float3* img_out, float3* img_in, float3* min, float3* max, uint32_t height, uint32_t width)
{
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t idx = y * width + x;

	if (x >= width || y >= height)
		return;

	img_out[idx] = (img_in[idx] - min[0]) / (max[0] - min[0]);
}

__global__ void write_srgb(cudaSurfaceObject_t surface, float3* img_in, float gamma, uint32_t height, uint32_t width)
{
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t idx = y * width + x;

    if (x >= width || y >= height)
        return;

    float inv_gamma = 1.0f / gamma;

    float3 rgb = img_in[idx];

    if (rgb.x < 0.0031308f)
        rgb.x = 12.92f * rgb.x;
    else
        rgb.x = 1.055f * powf(rgb.x, inv_gamma) - 0.055f;
    if (rgb.y < 0.0031308f)
        rgb.y = 12.92f * rgb.y;
    else
        rgb.y = 1.055f * powf(rgb.y, inv_gamma) - 0.055f;
    if (rgb.z < 0.0031308f)
        rgb.z = 12.92f * rgb.z;
	else
		rgb.z = 1.055f * powf(rgb.z, inv_gamma) - 0.055f;

    uchar4 rgba = make_uchar4(
        (uint8_t)(saturatef(rgb.x) * 255.0f),
        (uint8_t)(saturatef(rgb.y) * 255.0f),
        (uint8_t)(saturatef(rgb.z) * 255.0f),
        255
    );

    surf2Dwrite(rgba, surface, x * sizeof(uchar4), y);
}

__global__ void write_rgb(cudaSurfaceObject_t surface, float3* img_in, uint32_t height, uint32_t width)
{
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t idx = y * width + x;

	if (x >= width || y >= height)
		return;

	float3 rgb = img_in[idx];

	uchar4 rgba = make_uchar4(
		(uint8_t)(saturatef(rgb.x) * 255.0f),
		(uint8_t)(saturatef(rgb.y) * 255.0f),
		(uint8_t)(saturatef(rgb.z) * 255.0f),
		255
	);

	surf2Dwrite(rgba, surface, x * sizeof(uchar4), y);
}

__global__ void write_gray(cudaSurfaceObject_t surface, float* img_in, uint32_t height, uint32_t width)
{
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t idx = y * width + x;

    if (x >= width || y >= height)
        return;

    float gray = img_in[idx];

    uchar4 rgba = make_uchar4(
        (uint8_t)(saturatef(gray) * 255.0f),
        (uint8_t)(saturatef(gray) * 255.0f),
        (uint8_t)(saturatef(gray) * 255.0f),
        255
    );

    surf2Dwrite(rgba, surface, x * sizeof(uchar4), y);
}

static bool has_framebuffer_resized = false;

void glfw_framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	glViewport(0, 0, width, height);
	has_framebuffer_resized = true;
}

int __main()
{
    uint32_t img_width = 960;
    uint32_t img_height = 720;

    Camera camera = Camera(
        make_float3(0.0f, -10.0f, 2.5f),
        -90.0f, 0.0f,
        36.0f,
        (float)img_width / img_height,
        10.0f,
        4.0f // 0.0f for a pinhole camera
    );

    Material materials[] = {
        Material{ float3_one, float3_zero, 0.3f, 1.0f }, // mat_white,
        Material{ float3 { 0.8f, 0.1f, 0.0f }, float3_zero, 0.3f, 0.5f }, // mat_orange
        Material{ float3 { 0.0f, 0.25f, 0.8f }, float3_zero, 0.3f, 0.5f }, // mat_blue
        Material{ float3 { 0.8f, 0.4f, 0.0f }, float3_zero, 0.5f, 1.0f }, // mat_yellow
        Material{ float3_one, float3_zero, 0.0f, 1.0f }, // mat_mirror
        Material{ float3_one, float3_one * 16.0f, 0.5f, 0.5f }, // mat_light
        Material{ float3_one, float3 {1.0f, 0.0f, 0.0f } * 16.0f, 0.5f, 0.5f }, // mat_light_red
        Material{ float3_one, float3 {0.0f, 1.0f, 0.0f } * 16.0f, 0.5f, 0.5f }, // mat_light_green
        Material{ float3_one, float3 {0.0f, 0.0f, 1.0f } * 16.0f, 0.5f, 0.5f } // mat_light_blue
    };    

    Cube cubes[] = {
        Cube { float3 { 0.0f, 0.0f, -0.5f }, float3 { 6.0f, 24.0f, 1.0f }, float3_zero, 0 }, // bottom
        Cube { float3 { 0.0f, 0.0f, 5.5f }, float3 { 6.0f, 24.0f, 1.0f }, float3_zero, 0 }, // top
        Cube { float3 { -3.0f, 0.0f, 2.5f }, float3 { 1.0f, 24.0f, 6.0f }, float3_zero, 1 }, // left
        Cube { float3 { 3.0f, 0.0f, 2.5f }, float3 { 1.0f, 24.0f, 6.0f }, float3_zero, 2 }, // right
        Cube { float3 { 0.0f, 3.0f, 2.5f }, float3 { 6.0f, 1.0f, 6.0f }, float3_zero, 0 }, // back
        Cube { float3 { 0.0f, -10.5f, 2.5f }, float3 { 6.0f, 1.0f, 6.0f }, float3_zero, 0 }, // front
        Cube { float3 { 1.2f, -1.2f, 0.75f }, float3 { 1.5f, 1.5f, 1.5f }, float3 { 0.0f, 0.0f, -15.0f }, 3 }, // small cube
        Cube { float3 { -1.0f, 1.0f, 1.5f }, float3 { 1.75f, 1.75f, 3.0f }, float3 { 0.0f, 0.0f, 20.0f }, 4 }, // tall cube
        Cube { float3 { 1.6f, 1.6f, 4.9f }, float3 { 1.0f, 1.0f, 0.05f }, float3_zero, 6 }, // light 1
        Cube { float3 { 0.0f, 0.0f, 4.9f }, float3 { 1.0f, 1.0f, 0.05f }, float3 { 0.0f, 0.0f, 45.0f }, 7 }, // light 2
        Cube { float3 { -1.6f, -1.6f, 4.9f }, float3 { 1.0f, 1.0f, 0.05f }, float3_zero, 8 }, // light 3
        Cube { float3 { 0.0f, 0.0f, 2.5f }, float3 { 0.25f, 0.25f, 0.25f }, float3 { 45.0f, 45.0f, 45.0f }, 4 }
    };

    const uint32_t tri_count = sizeof(cubes) / sizeof(Cube) * 12;
    const uint32_t material_count = sizeof(materials) / sizeof(Material);

    Triangle tris[tri_count];
    std::vector<uint32_t> light_indices;
    for (uint32_t i = 0; i < sizeof(cubes) / sizeof(Cube); i++)
	{
		for (uint32_t j = 0; j < 12; j++)
		{
			tris[i * 12 + j] = cubes[i].tris[j];
            if (length(materials[cubes[i].material_id].emission) > 0.0f)
			{
				light_indices.push_back(i * 12 + j);
			}
		}
	}

    // Initialize CUDA
    CUDA_ERROR_CHECK(cudaSetDevice(0));

    // Initialize glew/glfw
    if (!glfwInit())
    {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return 1;
    }

    // Create a windowed mode window and its OpenGL context

    GLFWwindow* window = glfwCreateWindow((int)img_width, (int)img_height, "Insight", NULL, NULL);
    if (!window)
    {
        fprintf(stderr, "Failed to create GLFW window\n");
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, glfw_framebuffer_size_callback);
    glfwSwapInterval(0);

    GLenum gl_status = glewInit();
    if (gl_status != GLEW_OK)
	{
		fprintf(stderr, "Failed to initialize glew: %s\n", glewGetErrorString(gl_status));
		return 1;
	}

    // Set up ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    dim3 block_size(16, 16);
    dim3 grid_size;

    uint32_t block_size_flat = 16 * 16;
    uint32_t grid_size_flat;
    
    Scene scene;
    GBuffer gbuffer;

    float3 world_max = make_float3(64.0f);

    cudaStream_t stream;
    CUDA_ERROR_CHECK(cudaStreamCreate(&stream));
    scene.init(camera, tris, tri_count, materials, material_count, -world_max, world_max, stream);
    CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
    CUDA_ERROR_CHECK(cudaStreamDestroy(stream));


    // Loop until the user closes the window
    bool use_denoiser = false;
    bool initialize = true;
    bool deinitalize = false;
    bool stop = false;

    double mouse_xpos, mouse_ypos;
    glfwGetCursorPos(window, &mouse_xpos, &mouse_ypos);

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
        auto time_start = std::chrono::high_resolution_clock::now();

        if (initialize)
        {
            glfwGetFramebufferSize(window, (int*)&img_width, (int*)&img_height);
            gbuffer.init(img_height, img_width);
            scene.camera.resize(gbuffer.width, gbuffer.height);

            grid_size = dim3((gbuffer.width + block_size.x - 1) / block_size.x, (gbuffer.height + block_size.y - 1) / block_size.y);
            grid_size_flat = (gbuffer.height * gbuffer.width + block_size_flat - 1) / block_size_flat;

            // Set initial values
            cudaStream_t stream;
            CUDA_ERROR_CHECK(cudaStreamCreate(&stream));

            set_seeds KERNEL_ARGS4(grid_size, block_size, 0, stream) (gbuffer);

            CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
            CUDA_ERROR_CHECK(cudaStreamDestroy(stream));

            initialize = false;
            deinitalize = false;
        }

        cudaStream_t stream;
        CUDA_ERROR_CHECK(cudaStreamCreate(&stream));

        // Map the framebuffer texture to a CUDA resource
        gbuffer.create_surfaces(stream);

        pathtrace KERNEL_ARGS4(grid_size, block_size, 0, stream) (gbuffer, scene);
        gbuffer.samples += 1;
        CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));

        if (use_denoiser)
        {
            denoise KERNEL_ARGS4(grid_size, block_size, 0, stream) (gbuffer);
            CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
        }

        reduce_minmax KERNEL_ARGS4(grid_size_flat, block_size_flat, 2 * block_size_flat * sizeof(float3), stream) (gbuffer.position_min, gbuffer.position_max, gbuffer.position, gbuffer.position, img_height * img_width * 3);
        reduce_minmax KERNEL_ARGS4(grid_size_flat, block_size_flat, 2 * block_size_flat * sizeof(float3), stream) (gbuffer.depth_min, gbuffer.depth_max, gbuffer.depth_norm, gbuffer.depth_norm, img_height * img_width);
        reduce_minmax KERNEL_ARGS4(grid_size_flat, block_size_flat, 2 * block_size_flat * sizeof(float3), stream) (gbuffer.normal_min, gbuffer.normal_max, gbuffer.normal, gbuffer.normal, img_height * img_width);
        CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
        reduce_minmax KERNEL_ARGS4(1, block_size_flat, 2 * block_size_flat * sizeof(float3), stream) (gbuffer.position_min, gbuffer.position_max, gbuffer.position_min, gbuffer.position_max, grid_size_flat);
        reduce_minmax KERNEL_ARGS4(1, block_size_flat, 2 * block_size_flat * sizeof(float3), stream) (gbuffer.depth_min, gbuffer.depth_max, gbuffer.depth_min, gbuffer.depth_max, grid_size_flat);
        reduce_minmax KERNEL_ARGS4(1, block_size_flat, 2 * block_size_flat * sizeof(float3), stream) (gbuffer.normal_min, gbuffer.normal_max, gbuffer.normal_min, gbuffer.normal_max, grid_size_flat);
        CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));

        normalize KERNEL_ARGS4(grid_size, block_size, 0, stream) (gbuffer.position_norm, gbuffer.position, gbuffer.position_min, gbuffer.position_max, img_height, img_width);
        normalize KERNEL_ARGS4(grid_size, block_size, 0, stream) (gbuffer.depth_norm, gbuffer.depth_norm, gbuffer.depth_min, gbuffer.depth_max, img_height, img_width);
        normalize KERNEL_ARGS4(grid_size, block_size, 0, stream) (gbuffer.normal_norm, gbuffer.normal, gbuffer.normal_min, gbuffer.normal_max, img_height, img_width);
        CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));

        write_srgb KERNEL_ARGS4(grid_size, block_size, 0, stream) (gbuffer.surfaces[GBuffer::DENOISED], gbuffer.denoised, 2.2f, gbuffer.height, gbuffer.width);
        write_srgb KERNEL_ARGS4(grid_size, block_size, 0, stream) (gbuffer.surfaces[GBuffer::RADIANCE], gbuffer.radiance, 2.2f, gbuffer.height, gbuffer.width);
        write_rgb KERNEL_ARGS4(grid_size, block_size, 0, stream) (gbuffer.surfaces[GBuffer::POSITION], gbuffer.position_norm, gbuffer.height, gbuffer.width);
        write_rgb KERNEL_ARGS4(grid_size, block_size, 0, stream) (gbuffer.surfaces[GBuffer::DEPTH], gbuffer.depth_norm, gbuffer.height, gbuffer.width);
        write_rgb KERNEL_ARGS4(grid_size, block_size, 0, stream) (gbuffer.surfaces[GBuffer::NORMAL], gbuffer.normal_norm, gbuffer.height, gbuffer.width);
        write_rgb KERNEL_ARGS4(grid_size, block_size, 0, stream) (gbuffer.surfaces[GBuffer::ALBEDO], gbuffer.albedo, gbuffer.height, gbuffer.width);
        write_rgb KERNEL_ARGS4(grid_size, block_size, 0, stream) (gbuffer.surfaces[GBuffer::EMISSION], gbuffer.emission, gbuffer.height, gbuffer.width);
        write_gray KERNEL_ARGS4(grid_size, block_size, 0, stream) (gbuffer.surfaces[GBuffer::ROUGHNESS], gbuffer.roughness, gbuffer.height, gbuffer.width);
        write_gray KERNEL_ARGS4(grid_size, block_size, 0, stream) (gbuffer.surfaces[GBuffer::METALNESS], gbuffer.metalness, gbuffer.height, gbuffer.width);

        CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));

        // Blit the framebuffer texture to the window
        gbuffer.blit();
        gbuffer.frame_idx += 1;

        size_t vram_total_b, vram_free_b;
        CUDA_ERROR_CHECK(cudaMemGetInfo(&vram_free_b, &vram_total_b));
        float vram_used_mb = (vram_total_b - vram_free_b) / (1024.0f * 1024.0f);
        float vram_total_mb = vram_total_b / (1024.0f * 1024.0f);

        auto time_end = std::chrono::high_resolution_clock::now();
        float frametime = std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start).count() / 1000.0f;
        printf("%6d | %dx%d | %6f FPS | %6f ms | %2f/%2f \r", gbuffer.frame_idx, img_width, img_height, 1000.0f / frametime, frametime, vram_used_mb, vram_total_mb);

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        bool camera_updated = false;
        bool save_image = false;

        {
            ImGui::Begin("Controls");
            ImGui::Text("WASD: Move camera");
            ImGui::Text("Hold Shift: Move faster");
            ImGui::Text("Hold Middle Mouse: Look around");
            if (ImGui::Button("Save Render"))
			{
				save_image = true;
			}
            ImGui::End();
        }

        {
            ImGui::Begin("Performance");
            ImGui::Text("Sample: %d", gbuffer.samples);
            ImGui::Text("Frame: %d", gbuffer.frame_idx);
            ImGui::Text("Resolution: %dx%d", gbuffer.width, gbuffer.height);
            ImGui::Text("FPS: %.2f", 1000.0f / frametime);
            ImGui::Text("Frame time: %.2f ms", frametime);
            ImGui::Text("VRAM: %.2f/%.2f MB", vram_used_mb, vram_total_mb);
            ImGui::Checkbox("Use Denoiser", &use_denoiser);
            if (ImGui::Button("Reset"))
			{
				gbuffer.samples = 0;
			}
            ImGui::End();
        }

        {
            ImGui::Begin("Camera");
            ImGui::Text("Position: (%.2f, %.2f, %.2f)", scene.camera.position.x, scene.camera.position.y, scene.camera.position.z);
            ImGui::Text("Yaw/Pitch: %.2f, %.2f", scene.camera.yaw, scene.camera.pitch);
            ImGui::Text("Aspect ratio: %.2f", scene.camera.aspect_ratio);
            if (ImGui::SliderFloat("Sensor Width", &scene.camera.sensor_width, 1.0f, 100.0f, "%.2f"))
            {
                camera_updated = true;
            }
            if (ImGui::SliderFloat("Focal Distance", &scene.camera.focal_distance, 1.0f, 100.0f, "%.2f"))
            {
                camera_updated = true;
            }
            if (ImGui::SliderFloat("Aperture", &scene.camera.aperture, 0.0f, 32.0f, "f/%.2f"))
            {
                camera_updated = true;
            }
            ImGui::End();
        }

        {
            ImGui::Begin("Scene");
            ImGui::Text("Triangle count: %d", scene.tri_count);
            ImGui::Text("Light triangle count: %d", scene.light_count);
            ImGui::End();
        }

        {
            ImGui::Begin("GBuffer");
            ImVec2 size = ImVec2(128 * scene.camera.aspect_ratio, 128);
            ImGui::Text("Radiance");
            ImGui::Image((void*)(intptr_t)gbuffer.gl_textures[GBuffer::RADIANCE], size);
            ImGui::Text("Position");
            ImGui::Image((void*)(intptr_t)gbuffer.gl_textures[GBuffer::POSITION], size);
            ImGui::Text("Depth");
            ImGui::Image((void*)(intptr_t)gbuffer.gl_textures[GBuffer::DEPTH], size);
            ImGui::Text("Normal");
            ImGui::Image((void*)(intptr_t)gbuffer.gl_textures[GBuffer::NORMAL], size);
            ImGui::Text("Albedo");
            ImGui::Image((void*)(intptr_t)gbuffer.gl_textures[GBuffer::ALBEDO], size);
            ImGui::Text("Emission");
            ImGui::Image((void*)(intptr_t)gbuffer.gl_textures[GBuffer::EMISSION], size);
            ImGui::Text("Roughness/Metalness");
            ImGui::Image((void*)(intptr_t)gbuffer.gl_textures[GBuffer::ROUGHNESS], size);
            ImGui::Image((void*)(intptr_t)gbuffer.gl_textures[GBuffer::METALNESS], size);
            ImGui::End();
        }

        {
            ImGui::Begin("ImGui");
            if (ImGui::Button("Save Layout"))
			{
				ImGui::SaveIniSettingsToDisk("imgui.ini");
			}
            if (ImGui::Button("Load Layout"))
            {
                ImGui::LoadIniSettingsFromDisk("imgui.ini");
            }
            ImGui::End();
        }

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);

        // Mouse input
        double tmp_xpos = mouse_xpos;
        double tmp_ypos = mouse_ypos;
        glfwGetCursorPos(window, &tmp_xpos, &tmp_ypos);

        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_3) == GLFW_PRESS)
        {
            if (abs(mouse_xpos - tmp_xpos) > 0.1f || abs(mouse_ypos - tmp_ypos) > 0.1f)
            {
                float delta_x = (float)(mouse_xpos - tmp_xpos) * 0.2f;
                float delta_y = (float)(tmp_ypos - mouse_ypos) * 0.2f;

                scene.camera.yaw += delta_x;
                scene.camera.pitch += delta_y;

                mouse_xpos = tmp_xpos;
                mouse_ypos = tmp_ypos;
                camera_updated = true;
            }
        }
        else
        {
            mouse_xpos = tmp_xpos;
			mouse_ypos = tmp_ypos;
        }

        float3 movement = float3_zero;
        float camera_speed = 0.05f;
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        {
            movement.x -= 1.0f;
        }
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        {
            movement.x += 1.0f;
        }
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        {
            movement.y -= 1.0f;
        }
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        {
            movement.y += 1.0f;
        }
        if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
        {
            movement.z -= 1.0f;
        }
        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
        {
            movement.z += 1.0f;
        }

        if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
        {
            camera_speed *= 2.0f;
        }

        if (length(movement) > 0.0f)
        {
            movement = normalize(movement) * camera_speed;
            scene.camera.position += (scene.camera.right * movement.x) + (scene.camera.direction * movement.y) + (scene.camera.up * movement.z);
            camera_updated = true;
        }

        if (camera_updated)
        {
            scene.camera.update_variables();
            gbuffer.samples = 0;
        }

        if (save_image)
        {
            save_image = false;
            // save image
            uchar4* img_out = (uchar4*)malloc(gbuffer.width * gbuffer.height * sizeof(uchar4));
            
            // copy from the surface since its already converted to sRGB and uint8
            CUDA_ERROR_CHECK(cudaMemcpy2DFromArray(img_out, gbuffer.width * 4, gbuffer.cuda_arrays[GBuffer::DENOISED], 0, 0, gbuffer.width * 4, gbuffer.height, cudaMemcpyDeviceToHost));

            std::string filename = "output_" + std::to_string(gbuffer.frame_idx) + ".png";
            stbi_write_png(filename.c_str(), gbuffer.width, gbuffer.height, 4, img_out, gbuffer.width * 4);
            free(img_out);

            printf("\nSaved image at frame %d\r", gbuffer.frame_idx);
            // move console cursor up
            printf("\x1b[A\r");
        }

        CUDA_ERROR_CHECK(cudaGetLastError());
        CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
        gbuffer.destroy_surfaces(stream);

        CUDA_ERROR_CHECK(cudaStreamDestroy(stream));

        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        {
            deinitalize = true;
            stop = true;
        }

        if (has_framebuffer_resized)
        {
            deinitalize = true;
        }

        if (deinitalize)
		{
            initialize = true;
            has_framebuffer_resized = false;

            gbuffer.free();
 		}

        if (stop)
        {
            break;
        }
    }

    scene.free();

    CUDA_ERROR_CHECK(cudaDeviceReset());

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}

int main()
{
	try
	{
		return __main();
	}
	catch (const std::exception& e)
	{
		fprintf(stderr, "%s\n", e.what());
		return 1;
	}
}