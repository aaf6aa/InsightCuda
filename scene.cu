#pragma once

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>

#include "util.cu"
#include "math.cu"
#include "primitives.cu"
#include "camera.cu"

//#define USE_BVH
#ifdef USE_BVH
#include "bvh.cu"
#endif

__global__ void resolve_lights(uint32_t* light_indices, uint32_t* light_count, Triangle* tris, uint32_t tri_count, Material* materials, uint32_t material_count)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= tri_count)
	{
		return;
	}

	if (length(materials[tris[idx].material_id].emission) > 0.0f)
	{
		uint32_t light_idx = atomicAdd(light_count, 1);
		light_indices[light_idx] = idx;
	}
}

struct Scene
{
    Camera camera;
#ifdef USE_BVH
    BVH bvh;
#else
    Triangle* tris = nullptr;
#endif
    uint32_t tri_count = 0;
    uint32_t* lights = nullptr;
    uint32_t light_count = 0;
    Material* materials = nullptr;
    uint32_t material_count = 0;

    void init(Camera camera, Triangle* tris, uint32_t tri_count, Material* materials, uint32_t material_count, float3 world_min, float3 world_max, cudaStream_t& stream)
    {
        this->camera = camera;
        this->tri_count = tri_count;
        this->material_count = material_count;
        CUDA_ERROR_CHECK(cudaMalloc(&this->materials, material_count * sizeof(Material)));
        CUDA_ERROR_CHECK(cudaMemcpy(this->materials, materials, material_count * sizeof(Material), cudaMemcpyHostToDevice));
        
#ifdef USE_BVH
        this->bvh.init(tris, tri_count, world_min, world_max, stream);

        Triangle* d_tris = this->bvh.tris;
#else
        CUDA_ERROR_CHECK(cudaMalloc(&this->tris, tri_count * sizeof(Triangle)));
        CUDA_ERROR_CHECK(cudaMemcpy(this->tris, tris, tri_count * sizeof(Triangle), cudaMemcpyHostToDevice));

        Triangle* d_tris = this->tris;
#endif

        // find triangles that are emissive
        CUDA_ERROR_CHECK(cudaMalloc(&lights, tri_count * sizeof(uint32_t)));
        uint32_t* d_light_count;
        CUDA_ERROR_CHECK(cudaMalloc(&d_light_count, sizeof(uint32_t)));
        CUDA_ERROR_CHECK(cudaMemset(d_light_count, 0, sizeof(uint32_t)));

        uint32_t block_size = 16 * 16;
        uint32_t grid_size = (tri_count + block_size - 1) / block_size;
        resolve_lights KERNEL_ARGS4(grid_size, block_size, 0, stream) (lights, d_light_count, d_tris, tri_count, this->materials, material_count);
        CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));

        CUDA_ERROR_CHECK(cudaMemcpy(&light_count, d_light_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_ERROR_CHECK(cudaFree(d_light_count));
    }

    __device__ bool intersect(HitRecord& hit_out, Ray ray, float max_t = FLT_MAX)
    {
#ifdef USE_BVH
        return bvh.intersect(hit_out, ray, max_t);
#else
        HitRecord temp_hit;
        bool hit_anything = false;
        for (int i = 0; i < tri_count; i++)
        {
            if (tris[i].intersect(temp_hit, ray) && (!hit_anything || temp_hit.distance < hit_out.distance))
            {
                hit_anything = true;
                hit_out = temp_hit;
                hit_out.triangle_id = i;
            }
        }
        return hit_anything;
#endif
    }

    void free()
    {
#ifdef USE_BVH
        bvh.free();
#else
        CUDA_ERROR_CHECK(cudaFree(tris));
#endif
        CUDA_ERROR_CHECK(cudaFree(lights));
        CUDA_ERROR_CHECK(cudaFree(materials));
    }
};