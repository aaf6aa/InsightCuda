#pragma once

#include <float.h>
#include <math.h>
#include <stdint.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include "math.cu"

struct Ray
{
    float3 origin;
    float3 direction;
    float3 direction_inv;

    __host__ __device__ Ray(float3 origin, float3 direction)
        : origin(origin), direction(direction)
    {
        direction_inv = 1.0f / direction;
    }
};

struct Material
{
    float3 albedo;
    float3 emission;
    float roughness;
    float metalness;
    float reflectance = 0.5f;
};

struct HitRecord
{
    float3 position;
    float distance;
    float3 normal;
    uint32_t triangle_id;
    uint32_t material_id;
};

struct Triangle
{
    float3 vertices[3];
    float3 centroid;
    uint32_t material_id;

    __host__ __device__ Triangle()
    {
        vertices[0] = float3_zero;
		vertices[1] = float3_zero;
		vertices[2] = float3_zero;
		centroid = float3_zero;
        material_id = 0;
    }

    __host__ __device__ Triangle(float3 v0, float3 v1, float3 v2, uint32_t mat_id = 0)
	{
		vertices[0] = v0;
		vertices[1] = v1;
		vertices[2] = v2;
		centroid = (v0 + v1 + v2) / 3.0f;
        material_id = mat_id;
	}

    __host__ __device__ void bounding_box(float3& min, float3& max)
	{
		min = fminf(fminf(vertices[0], vertices[1]), vertices[2]);
		max = fmaxf(fmaxf(vertices[0], vertices[1]), vertices[2]);
	}

    __host__ __device__ bool intersect(HitRecord& hit, const Ray& ray)
    {
        float3 e1 = vertices[1] - vertices[0];
        float3 e2 = vertices[2] - vertices[0];
        float3 h = cross(ray.direction, e2);
        float a = dot(e1, h);

        if (fabs(a) < EPS_F)
        {
            return false;
        }

        float f = 1.0f / a;
        float3 s = ray.origin - vertices[0];
        float u = f * dot(s, h);

        if (u < 0.0f || u > 1.0f)
        {
            return false;
        }

        float3 q = cross(s, e1);
        float v = f * dot(ray.direction, q);

        if (v < 0.0f || u + v > 1.0f)
        {
            return false;
        }

        float t = f * dot(e2, q);

        if (t <= EPS_F)
            return false;

        hit.distance = t;
        hit.position = ray.origin + ray.direction * t;
        hit.normal = normalize(cross(e1, e2));
        // make sure normal is facing the ray
        if (dot(hit.normal, ray.direction) > 0.0f)
            hit.normal = -hit.normal;
        hit.material_id = material_id;

        return true;
    }
};

struct Cube
{
    float3 position;
    float3 size;
    float3 rotation; // XYZ euler
    uint32_t material_id;

    Triangle tris[12];

    __host__ __device__ Cube(float3 pos, float3 s, float3 rot, uint32_t mat_id)
        : position(pos), size(s), rotation(rot), material_id(mat_id)
    {
        update();
    }

    __host__ __device__ void set_position(float3 pos)
    {
        position = pos;
        update();
    }

    __host__ __device__ void set_size(float3 s)
    {
        size = s;
        update();
    }

    __host__ __device__ void set_rotation(float3 rot)
    {
        rotation = rot;
        update();
    }

    __host__ __device__ void update()
    {
        float3 half_size = size * 0.5f;

        float3 vertices[8] = {
            make_float3(-half_size.x, -half_size.y, -half_size.z),
            make_float3(half_size.x, -half_size.y, -half_size.z),
            make_float3(-half_size.x, half_size.y, -half_size.z),
            make_float3(half_size.x, half_size.y, -half_size.z),
            make_float3(-half_size.x, -half_size.y, half_size.z),
            make_float3(half_size.x, -half_size.y, half_size.z),
            make_float3(-half_size.x, half_size.y, half_size.z),
            make_float3(half_size.x, half_size.y, half_size.z)
        };

        // rotation
        // convert to radians
        float3 rot = make_float3(rotation.x * PI_F / 180.0f, rotation.y * PI_F / 180.0f, rotation.z * PI_F / 180.0f);
        float3 c = make_float3(cosf(rot.x), cosf(rot.y), cosf(rot.z));
        float3 s = make_float3(sinf(rot.x), sinf(rot.y), sinf(rot.z));

        float rotation_matrix[3][3] = {
            { c.z * c.y, c.z * s.y * s.x - s.z * c.x, c.z * s.y * c.x + s.z * s.x },
            { s.z * c.y, s.z * s.y * s.x + c.z * c.x, s.z * s.y * c.x - c.z * s.x },
            { -s.y,c.y * s.x, c.y * c.x }
        };

        for (uint8_t i = 0; i < 8; i++)
        {
            float3 vertex = vertices[i];
            vertices[i] = make_float3(
                rotation_matrix[0][0] * vertex.x + rotation_matrix[0][1] * vertex.y + rotation_matrix[0][2] * vertex.z,
                rotation_matrix[1][0] * vertex.x + rotation_matrix[1][1] * vertex.y + rotation_matrix[1][2] * vertex.z,
                rotation_matrix[2][0] * vertex.x + rotation_matrix[2][1] * vertex.y + rotation_matrix[2][2] * vertex.z
            ) + position;
        }

        const uint3 tri_indices[12] = {
            { 0, 1, 2 }, { 1, 3, 2 },
            { 4, 6, 5 }, { 5, 6, 7 },
            { 0, 4, 1 }, { 1, 4, 5 },
            { 2, 3, 6 }, { 3, 7, 6 },
            { 0, 2, 4 }, { 2, 6, 4 },
            { 1, 5, 3 }, { 3, 5, 7 }
        };

        // construct triangles
        for (uint32_t i = 0; i < 12; i++)
        {
            tris[i] = Triangle(vertices[tri_indices[i].x], vertices[tri_indices[i].y], vertices[tri_indices[i].z], material_id);
        }
    }
};