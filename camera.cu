#pragma once

#include <float.h>
#include <math.h>
#include <stdint.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include "math.cu"
#include "primitives.cu"

struct Camera
{
    float3 position = make_float3(5.0f, 5.0f, 5.0f);
    float yaw = -90.0f;
    float pitch = 0.0f;
    float sensor_width = 35.0f;
    float aspect_ratio = 1.77777778f;
    float focal_distance = 10.0f;
    float aperture = 2.2f;
    float lens_radius = 0.0f;
    float3 direction = float3_zero;
    float3 right = float3_zero;
    float3 up = float3_zero;
    float3 upper_left_corner = float3_zero;
    float3 horizontal = float3_zero;
    float3 vertical = float3_zero;

    __host__ __device__ Camera()
    {
        update_variables();
    }

    __host__ __device__ Camera(float3 position, float yaw, float pitch, float sensor_width, float aspect_ratio, float focal_distance, float aperture)
        : position(position), yaw(yaw), pitch(pitch), sensor_width(sensor_width), aspect_ratio(aspect_ratio), focal_distance(focal_distance), aperture(aperture)
    {
        update_variables();
    }

    __host__ __device__ void resize(uint32_t width, uint32_t height)
    {
        aspect_ratio = (float)width / (float)height;
        update_variables();
    }

    __host__ __device__ void update_variables()
    {
        float half_width = tanf(sensor_width * 0.75f * PI_F / 180.0f);
        float half_height = half_width / aspect_ratio;

        lens_radius = 0.0f;
        if (aperture > 0.0f)
        {
            lens_radius = (1.0f / aperture) / 2.0f;
        }

        // prevent gimbal lock
        if (pitch > 89.0f)
        {
            pitch = 89.0f;
        }
        if (pitch < -89.0f)
        {
            pitch = -89.0f;
        }

        float c_pitch = cosf(pitch * PI_F / 180.0f);
        float s_pitch = sinf(pitch * PI_F / 180.0f);
        float c_yaw = cosf(yaw * PI_F / 180.0f);
        float s_yaw = sinf(yaw * PI_F / 180.0f);

        direction = normalize(make_float3(
            c_yaw * c_pitch,
            s_yaw * c_pitch,
            s_pitch
        ));

        right = normalize(cross(float3_up, direction));
        up = cross(direction, right);

        upper_left_corner = position - right * half_width * focal_distance + up * half_height * focal_distance - direction * focal_distance;
        horizontal = right * 2.0f * half_width * focal_distance;
        vertical = up * 2.0f * half_height * focal_distance;
    }

    __host__ __device__ Ray generate_ray(float x, float y, uint32_t& seed) const
    {
        float3 offset = float3_zero;
        if (lens_radius > 0.0f && seed != 0)
        {
            float3 random_in_lens = random_in_unit_hemisphere(direction, seed) * lens_radius;
            offset = right * random_in_lens.x + up * random_in_lens.y;
        }

        float3 origin = position + offset;
        float3 direction = upper_left_corner + horizontal * x - vertical * y - position - offset;

        return Ray(origin, normalize(direction));
    }
};