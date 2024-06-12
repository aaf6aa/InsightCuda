#pragma once

#include <float.h>
#include <math.h>
#include <stdint.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#define MATH_FUNC static __inline__ __host__ __device__

#define EPS_F 1e-4f
#define EPS 1e-6f

#ifndef PI_F
#define PI_F CUDART_PI_F
#endif

#ifndef TWO_PI_F
#define TWO_PI_F CUDART_PI_F * 2.0f
#endif

#ifndef ONE_OVER_PI_F
#define ONE_OVER_PI_F (1.0f / CUDART_PI_F)
#endif

#ifndef ONE_OVER_TWO_PI_F
#define ONE_OVER_TWO_PI_F (1.0f / TWO_PI_F)
#endif

MATH_FUNC float saturatef(float x)
{
	return fminf(fmaxf(x, 0.0f), 1.0f);
}

MATH_FUNC uint32_t xorshift32s(uint32_t& seed)
{
	uint32_t x = seed;
	x ^= x >> 12;
	x ^= x << 25;
	x ^= x >> 27;
	seed = x;
	return x * 0x4F6CDD1DU;
}

MATH_FUNC uint64_t xorshift64s(uint64_t& seed)
{
	uint64_t x = seed;
	x ^= x >> 12;
	x ^= x << 25;
	x ^= x >> 27;
	seed = x;
	return x * 0x2545F4914F6CDD1DULL;
}

MATH_FUNC float xorshift32sf(uint32_t& seed)
{
	uint32_t x = xorshift32s(seed);
	return (float)(x >> 9) / (1u << 23);
}

MATH_FUNC float xorshift64sf(uint64_t& seed)
{
	uint64_t x = xorshift64s(seed);
	return (float)(x >> 11) / (1ull << 53);
}

MATH_FUNC float lerp(float a, float b, float t)
{
	return a + (b - a) * t;
}

#pragma region float2

#define float2_up float2 { 0.0f, 0.0f, 1.0f }
#define float2_zero float2 { 0.0f, 0.0f, 0.0f }
#define float2_one float2 { 1.0f, 1.0f, 1.0f }

MATH_FUNC float2 make_float2(float x)
{
	return make_float2(x, x);
}

MATH_FUNC float2 operator-(const float2& a)
{
	return make_float2(-a.x, -a.y);
}

MATH_FUNC float2 operator+(const float2& a, const float2& b)
{
	return make_float2(a.x + b.x, a.y + b.y);
}

MATH_FUNC void operator+=(float2& a, const float2& b)
{
	a.x += b.x;
	a.y += b.y;
}

MATH_FUNC float2 operator+(const float2& a, const float b)
{
	return make_float2(a.x + b, a.y + b);
}

MATH_FUNC void operator+=(float2& a, const float b)
{
	a.x += b;
	a.y += b;
}

MATH_FUNC float2 operator+(const float a, const float2& b)
{
	return make_float2(a + b.x, a + b.y);
}

MATH_FUNC float2 operator-(const float2& a, const float2& b)
{
	return make_float2(a.x - b.x, a.y - b.y);
}

MATH_FUNC void operator-=(float2& a, const float2& b)
{
	a.x -= b.x;
	a.y -= b.y;
}

MATH_FUNC float2 operator-(const float2& a, const float b)
{
	return make_float2(a.x - b, a.y - b);
}

MATH_FUNC void operator-=(float2& a, const float b)
{
	a.x -= b;
	a.y -= b;
}

MATH_FUNC float2 operator-(const float a, const float2& b)
{
	return make_float2(a - b.x, a - b.y);
}

MATH_FUNC float2 operator*(const float2& a, const float2& b)
{
	return make_float2(a.x * b.x, a.y * b.y);
}

MATH_FUNC void operator *=(float2& a, const float2& b)
{
	a.x *= b.x;
	a.y *= b.y;
}

MATH_FUNC float2 operator*(const float2& a, const float b)
{
	return make_float2(a.x * b, a.y * b);
}

MATH_FUNC void operator*=(float2& a, const float b)
{
	a.x *= b;
	a.y *= b;
}

MATH_FUNC float2 operator*(const float a, const float2& b)
{
	return make_float2(a * b.x, a * b.y);
}

MATH_FUNC float2 operator/(const float2& a, const float2& b)
{
	return make_float2(a.x / b.x, a.y / b.y);
}

MATH_FUNC void operator /=(float2& a, const float2& b)
{
	a.x /= b.x;
	a.y /= b.y;
}

MATH_FUNC float2 operator/(const float2& a, const float b)
{
	return make_float2(a.x / b, a.y / b);
}

MATH_FUNC void operator /=(float2& a, const float b)
{
	a.x /= b;
	a.y /= b;
}

MATH_FUNC float2 operator/(const float a, const float2& b)
{
	return make_float2(a / b.x, a / b.y);
}

MATH_FUNC float dot(const float2& a, const float2& b)
{
	return a.x * b.x + a.y * b.y;
}

MATH_FUNC float2 fabsf(const float2& a)
{
	return make_float2(fabsf(a.x), fabsf(a.y));
}

MATH_FUNC float2 fminf(const float2& a, const float2& b)
{
	return make_float2(fminf(a.x, b.x), fminf(a.y, b.y));
}

MATH_FUNC float2 fminf(const float2& a, const float b)
{
	return make_float2(fminf(a.x, b), fminf(a.y, b));
}

MATH_FUNC float2 fmaxf(const float2& a, const float2& b)
{
	return make_float2(fmaxf(a.x, b.x), fmaxf(a.y, b.y));
}

MATH_FUNC float2 fmaxf(const float2& a, const float b)
{
	return make_float2(fmaxf(a.x, b), fmaxf(a.y, b));
}

MATH_FUNC float length(const float2& a)
{
	return sqrtf(dot(a, a));
}

MATH_FUNC float length_squared(const float2& a)
{
	return dot(a, a);
}

MATH_FUNC float2 lerp(const float2& a, const float2& b, float t)
{
	return a + t * (b - a);
}

MATH_FUNC float2 lerp(const float2& a, const float2& b, const float2& t)
{
	return a + t * (b - a);
}

MATH_FUNC float2 normalize(const float2& a)
{
	float inv_length = rsqrtf(a.x * a.x + a.y * a.y);
	return a * inv_length;
}

MATH_FUNC float2 pow(const float2& a, float b)
{
	return make_float2(powf(a.x, b), powf(a.y, b));
}

MATH_FUNC float2 saturatef(const float2& a)
{
	return fminf(fmaxf(a, 0.0f), 1.0f);
}

#pragma endregion


#pragma region float3

#define float3_up float3 { 0.0f, 0.0f, 1.0f }
#define float3_zero float3 { 0.0f, 0.0f, 0.0f }
#define float3_one float3 { 1.0f, 1.0f, 1.0f }

MATH_FUNC float3 make_float3(float x)
{
	return make_float3(x, x, x);
}

MATH_FUNC float3 operator-(const float3& a)
{
	return make_float3(-a.x, -a.y, -a.z);
}

MATH_FUNC float3 operator+(const float3& a, const float3& b)
{
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

MATH_FUNC void operator+=(float3& a, const float3& b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
}

MATH_FUNC float3 operator+(const float3& a, const float b)
{
	return make_float3(a.x + b, a.y + b, a.z + b);
}

MATH_FUNC void operator+=(float3& a, const float b)
{
	a.x += b;
	a.y += b;
	a.z += b;
}

MATH_FUNC float3 operator+(const float a, const float3& b)
{
	return make_float3(a + b.x, a + b.y, a + b.z);
}

MATH_FUNC float3 operator-(const float3& a, const float3& b)
{
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

MATH_FUNC void operator-=(float3& a, const float3& b)
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
}

MATH_FUNC float3 operator-(const float3& a, const float b)
{
	return make_float3(a.x - b, a.y - b, a.z - b);
}

MATH_FUNC void operator-=(float3& a, const float b)
{
	a.x -= b;
	a.y -= b;
	a.z -= b;
}

MATH_FUNC float3 operator-(const float a, const float3& b)
{
	return make_float3(a - b.x, a - b.y, a - b.z);
}

MATH_FUNC float3 operator*(const float3& a, const float3& b)
{
	return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

MATH_FUNC void operator *=(float3& a, const float3& b)
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
}

MATH_FUNC float3 operator*(const float3& a, const float b)
{
	return make_float3(a.x * b, a.y * b, a.z * b);
}

MATH_FUNC void operator*=(float3& a, const float b)
{
	a.x *= b;
	a.y *= b;
	a.z *= b;
}

MATH_FUNC float3 operator*(const float a, const float3& b)
{
	return make_float3(a * b.x, a * b.y, a * b.z);
}

MATH_FUNC float3 operator/(const float3& a, const float3& b)
{
	return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

MATH_FUNC void operator /=(float3& a, const float3& b)
{
	a.x /= b.x;
	a.y /= b.y;
	a.z /= b.z;
}

MATH_FUNC float3 operator/(const float3& a, const float b)
{
	return make_float3(a.x / b, a.y / b, a.z / b);
}

MATH_FUNC void operator /=(float3& a, const float b)
{
	a.x /= b;
	a.y /= b;
	a.z /= b;
}

MATH_FUNC float3 operator/(const float a, const float3& b)
{
	return make_float3(a / b.x, a / b.y, a / b.z);
}

MATH_FUNC float3 cross(const float3& a, const float3& b)
{
	return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

MATH_FUNC float dot(const float3& a, const float3& b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

MATH_FUNC float3 fabsf(const float3& a)
{
	return make_float3(fabsf(a.x), fabsf(a.y), fabsf(a.z));
}

MATH_FUNC float3 fminf(const float3& a, const float3& b)
{
	return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

MATH_FUNC float3 fminf(const float3& a, const float b)
{
	return make_float3(fminf(a.x, b), fminf(a.y, b), fminf(a.z, b));
}

MATH_FUNC float3 fmaxf(const float3& a, const float3& b)
{
	return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

MATH_FUNC float3 fmaxf(const float3& a, const float b)
{
	return make_float3(fmaxf(a.x, b), fmaxf(a.y, b), fmaxf(a.z, b));
}

MATH_FUNC float length(const float3& a)
{
	return sqrtf(dot(a, a));
}

MATH_FUNC float length_squared(const float3& a)
{
	return dot(a, a);
}

MATH_FUNC float3 lerp(const float3& a, const float3& b, float t)
{
	return a + t * (b - a);
}

MATH_FUNC float3 lerp(const float3& a, const float3& b, const float3& t)
{
	return a + t * (b - a);
}

MATH_FUNC float3 normalize(const float3& a)
{
	float inv_length = rsqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
	return a * inv_length;
}

MATH_FUNC float3 pow(const float3& a, float b)
{
	return make_float3(powf(a.x, b), powf(a.y, b), powf(a.z, b));
}

MATH_FUNC float3 random_cosine_uintvec3(uint32_t& seed)
{
	// cosine weighted sphere sampling
	float theta = 2.0f * CUDART_PI_F * xorshift32sf(seed);
	float phi = acosf(2.0f * xorshift32sf(seed) - 1.0f);
	float r = cbrtf(xorshift32sf(seed));

	return make_float3(
		r * sinf(phi) * cosf(theta),
		r * sinf(phi) * sinf(theta),
		r * cosf(phi)
	);
}

MATH_FUNC float3 random_in_unit_hemisphere(const float3& normal, uint32_t& seed)
{
	float3 unitvec3 = random_cosine_uintvec3(seed);
	if (dot(unitvec3, normal) > 0.0f)
		return unitvec3;
	else
		return -unitvec3;
}

MATH_FUNC float3 reflect(const float3& v, const float3& n)
{
	return v - n * dot(v, n) * 2.0f;
}

MATH_FUNC float3 saturatef(const float3& a)
{
	return fminf(fmaxf(a, 0.0f), 1.0f);
}

#pragma endregion

#pragma region float4

#define float4_zero float4 { 0.0f, 0.0f, 0.0f, 0.0f }
#define float4_one float4 { 1.0f, 1.0f, 1.0f, 1.0f }

MATH_FUNC float4 make_float4(float x)
{
	return make_float4(x, x, x, x);
}

MATH_FUNC float4 operator-(const float4& a)
{
	return make_float4(-a.x, -a.y, -a.z, -a.w);
}

MATH_FUNC float4 operator+(const float4& a, const float4& b)
{
	return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

MATH_FUNC void operator+=(float4& a, const float4& b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	a.w += b.w;
}

MATH_FUNC float4 operator+(const float4& a, const float b)
{
	return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}

MATH_FUNC float4 operator-(const float4& a, const float4& b)
{
	return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

MATH_FUNC void operator-=(float4& a, const float4& b)
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	a.w -= b.w;
}

MATH_FUNC float4 operator-(const float4& a, const float b)
{
	return make_float4(a.x - b, a.y - b, a.z - b, a.w - b);
}

MATH_FUNC float4 operator*(const float4& a, const float4& b)
{
	return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

MATH_FUNC void operator*=(float4& a, const float4& b)
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
	a.w *= b.w;
}

MATH_FUNC float4 operator*(const float4& a, const float b)
{
	return make_float4(a.x * b, a.y * b, a.z * b, a.w - b);
}

MATH_FUNC void operator*=(float4& a, const float b)
{
	a.x *= b;
	a.y *= b;
	a.z *= b;
	a.w *= b;
}

MATH_FUNC float4 operator/(const float4& a, const float4& b)
{
	return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}

MATH_FUNC void operator/=(float4& a, const float4& b)
{
	a.x /= b.x;
	a.y /= b.y;
	a.z /= b.z;
	a.w /= b.w;
}

MATH_FUNC float4 operator/(const float4& a, const float b)
{
	return make_float4(a.x / b, a.y / b, a.z / b, a.w / b);
}

MATH_FUNC void operator/=(float4& a, const float b)
{
	a.x /= b;
	a.y /= b;
	a.z /= b;
	a.w /= b;
}

MATH_FUNC float dot(const float4& a, const float4& b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

MATH_FUNC float4 fabsf(const float4& a)
{
	return make_float4(fabsf(a.x), fabsf(a.y), fabsf(a.z), fabsf(a.w));
}

MATH_FUNC float4 fminf(const float4& a, const float4& b)
{
	return make_float4(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z), fminf(a.w, b.w));
}

MATH_FUNC float4 fminf(const float4& a, const float b)
{
	return make_float4(fminf(a.x, b), fminf(a.y, b), fminf(a.z, b), fminf(a.w, b));
}

MATH_FUNC float4 fmaxf(const float4& a, const float4& b)
{
	return make_float4(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z), fmaxf(a.w, b.w));
}

MATH_FUNC float4 fmaxf(const float4& a, const float b)
{
	return make_float4(fmaxf(a.x, b), fmaxf(a.y, b), fmaxf(a.z, b), fmaxf(a.w, b));
}

MATH_FUNC float length(const float4& a)
{
	return sqrtf(dot(a, a));
}

MATH_FUNC float4 lerp(const float4& a, const float4& b, float t)
{
	return a + (b - a) * t;
}

MATH_FUNC float4 lerp(const float4& a, const float4& b, const float4& t)
{
	return a + (b - a) * t;
}

MATH_FUNC float4 normalize(const float4& a)
{
	float inv_length = rsqrtf(dot(a, a));
	return a * inv_length;
}

MATH_FUNC float4 pow(const float4& a, float b)
{
	return make_float4(powf(a.x, b), powf(a.y, b), powf(a.z, b), powf(a.w, b));
}

MATH_FUNC float4 saturatef(const float4& a)
{
	return fminf(fmaxf(a, 0.0f), 1.0f);
}

#pragma endregion