/* Creative Commons CC0 Public Domain. To the extent possible under law, Jakub Boksansky has waived all copyright and related or neighboring rights to Crash Course in BRDF Implementation Code Sample. This work is published from: Germany. */
// This is a code sample accompanying the "Crash Course in BRDF Implementation" article
// v1.2, August 2023
// https://boksajak.github.io/blog/BRDF
// https://github.com/boksajak/brdf/blob/master/brdf.h
// Modified for CUDA by: Andrejs Krauze (https://github.com/aaf6aa)
// June 2024

#pragma once

#include <float.h>
#include <math.h>
#include <stdint.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include "math.cu"

namespace brdf {

#define BRDF_FUNCTION static __inline__ __host__ __device__

// -------------------------------------------------------------------------
//    Constant Definitions
// -------------------------------------------------------------------------

#define NONE 0

// NDF definitions
#define GGX 1
#define BECKMANN 2

// Specular BRDFs
#define MICROFACET 1
#define PHONG 2

// Diffuse BRDFs
#define LAMBERTIAN 1
#define OREN_NAYAR 2
#define DISNEY 3
#define FROSTBITE 4

// -------------------------------------------------------------------------
//    Configuration macros (user editable - set your preferences here)
// -------------------------------------------------------------------------

// Specify what NDF (GGX or BECKMANN you want to use)
#ifndef MICROFACET_DISTRIBUTION
#define MICROFACET_DISTRIBUTION GGX
//#define MICROFACET_DISTRIBUTION BECKMANN
#endif

// Specify default specular and diffuse BRDFs
#ifndef SPECULAR_BRDF
#define SPECULAR_BRDF MICROFACET
//#define SPECULAR_BRDF PHONG
//#define SPECULAR_BRDF NONE
#endif

#ifndef DIFFUSE_BRDF
//#define DIFFUSE_BRDF LAMBERTIAN
//#define DIFFUSE_BRDF OREN_NAYAR
//#define DIFFUSE_BRDF DISNEY
#define DIFFUSE_BRDF FROSTBITE
//#define DIFFUSE_BRDF NONE
#endif

// Specifies minimal reflectance for dielectrics (when metalness is zero)
// Nothing has lower reflectance than 2%, but we use 4% to have consistent results with UE4, Frostbite, et al.
// Note: only takes effect when USE_REFLECTANCE_PARAMETER is not defined
#define MIN_DIELECTRICS_F0 0.04f

// Define this to use minimal reflectance (F0) specified per material, instead of global MIN_DIELECTRICS_F0 value
#define USE_REFLECTANCE_PARAMETER 1

// Enable this to weigh diffuse by Fresnel too, otherwise specular and diffuse will be simply added together
// (this is disabled by default for Frostbite diffuse which is normalized to combine well with GGX Specular BRDF)
#if DIFFUSE_BRDF != FROSTBITE
#define COMBINE_BRDFS_WITH_FRESNEL 1
#endif

// Enable to calculate Fresnel term with spherical gaussian approximation
#define USE_GAUSSIAN_FRESNEL 1

// Uncomment this to use "general" version of G1 which is not optimized and uses NDF-specific G_Lambda (can be useful for experimenting and debugging)
//#define Smith_G1 Smith_G1_General 

// Enable optimized G2 implementation which includes division by specular BRDF denominator (not available for all NDFs, check macro G2_DIVIDED_BY_DENOMINATOR if it was actually used)
#define USE_OPTIMIZED_G2 1

// Enable height correlated version of G2 term. Separable version will be used otherwise
#define USE_HEIGHT_CORRELATED_G2 1

// Enable this to use Walter's sampling for GG-X distribution instead of more recent VNDF
//#define USE_WALTER_GGX_SAMPLING 1

// Enable this to VNDF sampling using spherical caps instead of original Heitz's method
#define USE_VNDF_WITH_SPHERICAL_CAPS 1

// -------------------------------------------------------------------------
//    Automatically resolved macros based on preferences (don't edit these)
// -------------------------------------------------------------------------

// Select distribution function
#if MICROFACET_DISTRIBUTION == GGX
#define Microfacet_D GGX_D
#elif MICROFACET_DISTRIBUTION == BECKMANN
#define Microfacet_D Beckmann_D
#endif

// Select G functions (masking/shadowing) depending on selected distribution
#if MICROFACET_DISTRIBUTION == GGX
#define Smith_G_Lambda Smith_G_Lambda_GGX
#elif MICROFACET_DISTRIBUTION == BECKMANN
#define Smith_G_Lambda Smith_G_Lambda_Beckmann_Walter
#endif

#ifndef Smith_G1
// Define version of G1 optimized specifically for selected NDF
#if MICROFACET_DISTRIBUTION == GGX
#define Smith_G1 Smith_G1_GGX
#elif MICROFACET_DISTRIBUTION == BECKMANN
#define Smith_G1 Smith_G1_Beckmann_Walter
#endif
#endif

// Select default specular and diffuse BRDF functions
#if SPECULAR_BRDF == MICROFACET
#define evalSpecular evalMicrofacet
#define sampleSpecular sampleSpecularMicrofacet
#if MICROFACET_DISTRIBUTION == GGX
#if USE_WALTER_GGX_SAMPLING
#define sampleSpecularHalfVector sampleGGXWalter
#else
#define sampleSpecularHalfVector sampleGGXVNDF
#endif
#else
#define sampleSpecularHalfVector sampleBeckmannWalter
#endif
#elif SPECULAR_BRDF == PHONG
#define evalSpecular evalPhong
#define sampleSpecular sampleSpecularPhong
#define sampleSpecularHalfVector samplePhong
#else
#define evalSpecular evalVoid
#define sampleSpecular sampleSpecularVoid
#define sampleSpecularHalfVector sampleSpecularHalfVectorVoid
#endif

#if MICROFACET_DISTRIBUTION == GGX
#if USE_WALTER_GGX_SAMPLING
#define specularSampleWeight specularSampleWeightGGXWalter
#else
#define specularSampleWeight specularSampleWeightGGXVNDF
#endif
#if USE_WALTER_GGX_SAMPLING
#define specularPdf sampleWalterReflectionPdf
#else
#define specularPdf sampleGGXVNDFReflectionPdf
#endif
#else
#define specularSampleWeight specularSampleWeightBeckmannWalter
#define specularPdf sampleWalterReflectionPdf
#endif

#if DIFFUSE_BRDF == LAMBERTIAN
#define evalDiffuse evalLambertian
#define diffuseTerm lambertian
#elif DIFFUSE_BRDF == OREN_NAYAR
#define evalDiffuse evalOrenNayar
#define diffuseTerm orenNayar
#elif DIFFUSE_BRDF == DISNEY
#define evalDiffuse evalDisneyDiffuse
#define diffuseTerm disneyDiffuse
#elif DIFFUSE_BRDF == FROSTBITE
#define evalDiffuse evalFrostbiteDisneyDiffuse
#define diffuseTerm frostbiteDisneyDiffuse
#else
#define evalDiffuse evalVoid
#define evalIndirectDiffuse evalIndirectVoid
#define diffuseTerm none
#endif

// -------------------------------------------------------------------------
//    Structures
// -------------------------------------------------------------------------

struct MaterialProperties
{
	float3 baseColor;
	float3 emissive;
	float roughness;
	float metalness;
	float reflectance;		//< This should default to 0.5 to set minimal reflectance at 4%
};

// Data needed to evaluate BRDF (surface and material properties at given point + configuration of light and normal vectors)
struct BrdfData
{
	// Material properties
	float3 specularF0;
	float3 diffuseReflectance;

	// Roughnesses
	float roughness;    //< perceptively linear roughness (artist's input)
	float alpha;        //< linear roughness - often 'alpha' in specular BRDF equations
	float alphaSquared; //< alpha squared - pre-calculated value commonly used in BRDF equations

	// Commonly used terms for BRDF evaluation
	float3 F; //< Fresnel term

	// Vectors
	float3 V; //< Direction to viewer (or opposite direction of incident ray)
	float3 N; //< Shading normal
	float3 H; //< Half vector (microfacet normal)
	float3 L; //< Direction to light (or direction of reflecting ray)

	float NdotL;
	float NdotV;

	float LdotH;
	float NdotH;
	float VdotH;

	// True when V/L is backfacing wrt. shading normal N
	bool Vbackfacing;
	bool Lbackfacing;
};

// -------------------------------------------------------------------------
//    Utilities
// -------------------------------------------------------------------------

// Converts Phong's exponent (shininess) to Beckmann roughness (alpha)
// Source: "Microfacet Models for Refraction through Rough Surfaces" by Walter et al.
BRDF_FUNCTION float shininessToBeckmannAlpha(float shininess)
{
	return sqrt(2.0f / (shininess + 2.0f));
}

// Converts Beckmann roughness (alpha) to Phong's exponent (shininess)
// Source: "Microfacet Models for Refraction through Rough Surfaces" by Walter et al.
BRDF_FUNCTION float beckmannAlphaToShininess(float alpha)
{
	return 2.0f / fminf(0.9999f, fmaxf(0.0002f, (alpha * alpha))) - 2.0f;
}

// Converts Beckmann roughness (alpha) to Oren-Nayar roughness (sigma)
// Source: "Moving Frostbite to Physically Based Rendering" by Lagarde & de Rousiers
BRDF_FUNCTION float beckmannAlphaToOrenNayarRoughness(float alpha)
{
	return 0.7071067f * atanf(alpha);
}

BRDF_FUNCTION float luminance(float3 rgb)
{
	return dot(rgb, make_float3(0.2126f, 0.7152f, 0.0722f));
}

BRDF_FUNCTION float3 baseColorToSpecularF0(const float3 baseColor, const float metalness, const float reflectance = 0.5f)
{
#if USE_REFLECTANCE_PARAMETER
	const float minDielectricsF0 = 0.16f * reflectance * reflectance;
#else
	const float minDielectricsF0 = MIN_DIELECTRICS_F0;
#endif
	return lerp(make_float3(minDielectricsF0), baseColor, metalness);
}

BRDF_FUNCTION float3 baseColorToDiffuseReflectance(float3 baseColor, float metalness)
{
	return baseColor * (1.0f - metalness);
}

BRDF_FUNCTION float none(const BrdfData data)
{
	return 0.0f;
}

BRDF_FUNCTION float3 evalVoid(const BrdfData data)
{
	return float3_zero;
}

BRDF_FUNCTION void evalIndirectVoid(const BrdfData data, float2 u, float3& rayDirection, float3& weight)
{
	rayDirection = float3_up;
	weight = float3_zero;
}

BRDF_FUNCTION float3 sampleSpecularVoid(float3 Vlocal, float alpha, float alphaSquared, float3 specularF0, float2 u, float3& weight)
{
	weight = float3_zero;
	return float3_zero;
}

BRDF_FUNCTION float3 sampleSpecularHalfVectorVoid(float3 Vlocal, float2 alpha2D, float2 u)
{
	return float3_zero;
}

// -------------------------------------------------------------------------
//    Quaternion rotations
// -------------------------------------------------------------------------

// Calculates rotation quaternion from input vector to the vector (0, 0, 1)
// Input vector must be normalized!
BRDF_FUNCTION float4 getRotationToZAxis(float3 input)
{
	// Handle special case when input is exact or near opposite of (0, 0, 1)
	if (input.z < -0.99999f) return make_float4(1.0f, 0.0f, 0.0f, 0.0f);
	
	//return normalize(make_float4(input.y, -input.x, 0.0f, 1.0f + input.z));

	float3 z_axis =  float3_up;
	float3 axis = cross(input, z_axis);
	float half_angle = acosf(dot(input, z_axis)) * 0.5f;
	float s = sinf(half_angle);

	return make_float4(axis.x * s, axis.y * s, axis.z * s, cosf(half_angle));
};

// Calculates rotation quaternion from vector (0, 0, 1) to the input vector
// Input vector must be normalized!
BRDF_FUNCTION float4 getRotationFromZAxis(float3 input)
{
	// Handle special case when input is exact or near opposite of (0, 0, 1)
	if (input.z < -0.99999f) return make_float4(1.0f, 0.0f, 0.0f, 0.0f);

	//return normalize(make_float4(-input.y, input.x, 0.0f, 1.0f + input.z));

	float3 z_axis = float3_up;
	float3 axis = cross(z_axis, input);
	float half_angle = acosf(dot(z_axis, input)) * 0.5f;
	float s = sinf(half_angle);

	return make_float4(axis.x * s, axis.y * s, axis.z * s, cosf(half_angle));
}

// Returns the quaternion with inverted rotation
BRDF_FUNCTION float4 invertRotation(float4 q)
{
	return make_float4(-q.x, -q.y, -q.z, q.w);
}

// Optimized point rotation using quaternion
// Source: https://gamedev.stackexchange.com/questions/28395/rotating-vector3-by-a-quaternion
BRDF_FUNCTION float3 rotatePoint(float4 q, float3 v)
{
	const float3 qAxis = make_float3(q.x, q.y, q.z);
	return 2.0f * dot(qAxis, v) * qAxis + (q.w * q.w - dot(qAxis, qAxis)) * v + 2.0f * q.w * cross(qAxis, v);
}

// -------------------------------------------------------------------------
//    Sampling
// -------------------------------------------------------------------------

// Samples a direction within a hemisphere oriented along +Z axis with a cosine-weighted distribution 
// Source: "Sampling Transformations Zoo" in Ray Tracing Gems by Shirley et al.
BRDF_FUNCTION float3 sampleHemisphere(float2 u, float& pdf)
{
	float a = sqrtf(u.x);
	float b = TWO_PI_F * u.y;

	float3 result = make_float3(
		a * cosf(b),
		a * sinf(b),
		sqrtf(1.0f - u.x));

	pdf = result.z * ONE_OVER_PI_F;

	return result;
}

BRDF_FUNCTION float3 sampleHemisphere(float2 u)
{
	float pdf;
	return sampleHemisphere(u, pdf);
}

// For sampling of all our diffuse BRDFs we use cosine-weighted hemisphere sampling, with PDF equal to (NdotL/PI)
BRDF_FUNCTION float diffusePdf(float NdotL)
{
	return NdotL * ONE_OVER_PI_F;
}

// -------------------------------------------------------------------------
//    Fresnel
// -------------------------------------------------------------------------

// Schlick's approximation to Fresnel term
// f90 should be 1.0, except for the trick used by Schuler (see 'shadowedF90' function)
BRDF_FUNCTION float3 evalFresnelSchlick(float3 f0, float f90, float NdotS)
{
	return f0 + (f90 - f0) * powf(1.0f - NdotS, 5.0f);
}

// Schlick's approximation to Fresnel term calculated using spherical gaussian approximation
// Source: https://seblagarde.wordpress.com/2012/06/03/spherical-gaussien-approximation-for-blinn-phong-phong-and-fresnel/ by Lagarde
BRDF_FUNCTION float3 evalFresnelSchlickSphericalGaussian(float3 f0, float f90, float NdotV)
{
	return f0 + (f90 - f0) * exp2f((-5.55473f * NdotV - 6.983146f) * NdotV);
}

// Schlick's approximation to Fresnel term with Hoffman's improvement using the Lazanyi's error term
// Source: "Fresnel Equations Considered Harmful" by Hoffman
// Also see slides http://renderwonk.com/publications/mam2019/naty_mam2019.pdf for examples and explanation of f82 term
BRDF_FUNCTION float3 evalFresnelHoffman(float3 f0, float f82, float f90, float NdotS)
{
	const float alpha = 6.0f; //< Fixed to 6 in order to put peak angle for Lazanyi's error term at 82 degrees (f82)
	float3 a = 17.6513846f * (f0 - f82) + 8.166666f * (float3_one - f0);
	return saturatef(f0 + (f90 - f0) * powf(1.0f - NdotS, 5.0f) - a * NdotS * powf(1.0f - NdotS, alpha));
}

BRDF_FUNCTION float3 evalFresnel(float3 f0, float f90, float NdotS)
{
#ifdef USE_GAUSSIAN_FRESNEL
	return evalFresnelSchlickSphericalGaussian(f0, f90, NdotS);
#else
	// Default is Schlick's approximation
	return evalFresnelSchlick(f0, f90, NdotS);
#endif
}

// Attenuates F90 for very low F0 values
// Source: "An efficient and Physically Plausible Real-Time Shading Model" in ShaderX7 by Schuler
// Also see section "Overbright highlights" in Hoffman's 2010 "Crafting Physically Motivated Shading Models for Game Development" for discussion
// IMPORTANT: Note that when F0 is calculated using metalness, it's value is never less than MIN_DIELECTRICS_F0, and therefore,
// this adjustment has no effect. To be effective, F0 must be authored separately, or calculated in different way. See main text for discussion.
BRDF_FUNCTION float shadowedF90(float3 F0)
{
	// This scaler value is somewhat arbitrary, Schuler used 60 in his article. In here, we derive it from MIN_DIELECTRICS_F0 so
	// that it takes effect for any reflectance lower than least reflective dielectrics
	//const float t = 60.0f;
	const float t = (1.0f / MIN_DIELECTRICS_F0);
	return fminf(1.0f, t * luminance(F0));
}

// -------------------------------------------------------------------------
//    Lambert
// -------------------------------------------------------------------------

BRDF_FUNCTION float lambertian(const BrdfData data)
{
	return 1.0f;
}

BRDF_FUNCTION float3 evalLambertian(const BrdfData data)
{
	return data.diffuseReflectance * (ONE_OVER_PI_F * data.NdotL);
}

// -------------------------------------------------------------------------
//    Phong
// -------------------------------------------------------------------------

// For derivation see "Phong Normalization Factor derivation" by Giesen
BRDF_FUNCTION float phongNormalizationTerm(float shininess)
{
	return (1.0f + shininess) * ONE_OVER_TWO_PI_F;
}

BRDF_FUNCTION float3 evalPhong(const BrdfData data)
{
	// First convert roughness to shininess (Phong exponent)
	float shininess = beckmannAlphaToShininess(data.alpha);

	float3 R = reflect(-data.L, data.N);
	return data.specularF0 * (phongNormalizationTerm(shininess) * powf(fmaxf(0.0f, dot(R, data.V)), shininess) * data.NdotL);
}

// Samples a Phong distribution lobe oriented along +Z axis
// Source: "Sampling Transformations Zoo" in Ray Tracing Gems by Shirley et al.
BRDF_FUNCTION float3 samplePhong(float3 Vlocal, float shininess, float2 u, float& pdf)
{
	float cosTheta = powf(1.0f - u.x, 1.0f / (1.0f + shininess));
	float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);

	float phi = TWO_PI_F * u.y;

	pdf = phongNormalizationTerm(shininess) * powf(cosTheta, shininess);

	return make_float3(
		cosf(phi) * sinTheta,
		sinf(phi) * sinTheta,
		cosTheta);
}

BRDF_FUNCTION float3 samplePhong(float3 Vlocal, float2 alpha2D, float2 u)
{
	float shininess = beckmannAlphaToShininess(dot(alpha2D, make_float2(0.5f)));
	float pdf;
	return samplePhong(Vlocal, shininess, u, pdf);
}

// Sampling the specular BRDF based on Phong, includes normalization term
BRDF_FUNCTION float3 sampleSpecularPhong(float3 Vlocal, float alpha, float alphaSquared, float3 specularF0, float2 u, float3& weight)
{
	// First convert roughness to shininess (Phong exponent)
	float shininess = beckmannAlphaToShininess(alpha);

	float pdf;
	float3 LPhong = samplePhong(Vlocal, shininess, u, pdf);

	// Sampled LPhong is in "lobe space" - where Phong lobe is centered around +Z axis
	// We need to rotate it in direction of perfect reflection
	float3 Nlocal = float3_up;
	float3 lobeDirection = reflect(-Vlocal, Nlocal);
	float3 Llocal = rotatePoint(getRotationFromZAxis(lobeDirection), LPhong);

	// Calculate the weight of the sample 
	float3 Rlocal = reflect(-Llocal, Nlocal);
	float NdotL = fmaxf(0.00001f, dot(Nlocal, Llocal));
	weight = fmaxf(float3_zero, specularF0 * NdotL);

	// Unoptimized formula was:
	//weight = specularF0 * (phongNormalizationTerm(shininess) * pow(max(0.0f, dot(Rlocal, Vlocal)), shininess) * NdotL) / pdf;

	return Llocal;
}

// -------------------------------------------------------------------------
//    Oren-Nayar 
// -------------------------------------------------------------------------

// Based on Oren-Nayar's qualitative model
// Source: "Generalization of Lambert's Reflectance Model" by Oren & Nayar
BRDF_FUNCTION float orenNayar(BrdfData data)
{
	// Oren-Nayar roughness (sigma) is in radians - use conversion from Beckmann roughness here
	float sigma = beckmannAlphaToOrenNayarRoughness(data.alpha);

	float thetaV = acosf(data.NdotV);
	float thetaL = acosf(data.NdotL);

	float alpha = fmaxf(thetaV, thetaL);
	float beta = fminf(thetaV, thetaL);

	// Calculate cosine of azimuth angles difference - by projecting L and V onto plane defined by N. Assume L, V, N are normalized.
	float3 l = data.L - data.NdotL * data.N;
	float3 v = data.V - data.NdotV * data.N;
	float cosPhiDifference = dot(normalize(v), normalize(l));

	float sigma2 = sigma * sigma;
	float A = 1.0f - 0.5f * (sigma2 / (sigma2 + 0.33f));
	float B = 0.45f * (sigma2 / (sigma2 + 0.09f));

	return (A + B * fmaxf(0.0f, cosPhiDifference) * sinf(alpha) * tanf(beta));
}

BRDF_FUNCTION float3 evalOrenNayar(const BrdfData data)
{
	return data.diffuseReflectance * (orenNayar(data) * ONE_OVER_PI_F * data.NdotL);
}

// -------------------------------------------------------------------------
//    Disney
// -------------------------------------------------------------------------

// Disney's diffuse term
// Source "Physically-Based Shading at Disney" by Burley
BRDF_FUNCTION float disneyDiffuse(const BrdfData data)
{
	float FD90MinusOne = 2.0f * data.roughness * data.LdotH * data.LdotH - 0.5f;

	float FDL = 1.0f + (FD90MinusOne * powf(1.0f - data.NdotL, 5.0f));
	float FDV = 1.0F + (FD90MinusOne * powf(1.0f - data.NdotV, 5.0f));

	return FDL * FDV;
}

BRDF_FUNCTION float3 evalDisneyDiffuse(const BrdfData data)
{
	return data.diffuseReflectance * (disneyDiffuse(data) * ONE_OVER_PI_F * data.NdotL);
}

// Frostbite's version of Disney diffuse with energy normalization.
// Source: "Moving Frostbite to Physically Based Rendering" by Lagarde & de Rousiers
BRDF_FUNCTION float frostbiteDisneyDiffuse(const BrdfData data)
{
	float energyBias = 0.5f * data.roughness;
	float energyFactor = lerp(1.0f, 1.0f / 1.51f, data.roughness);

	float FD90MinusOne = energyBias + 2.0f * data.LdotH * data.LdotH * data.roughness - 1.0f;

	float FDL = 1.0f + (FD90MinusOne * powf(1.0f - data.NdotL, 5.0f));
	float FDV = 1.0f + (FD90MinusOne * powf(1.0f - data.NdotV, 5.0f));

	return FDL * FDV * energyFactor;
}

BRDF_FUNCTION float3 evalFrostbiteDisneyDiffuse(const BrdfData data)
{
	return data.diffuseReflectance * (frostbiteDisneyDiffuse(data) * ONE_OVER_PI_F * data.NdotL);
}

// -------------------------------------------------------------------------
//    Smith G term
// -------------------------------------------------------------------------

// Function to calculate 'a' parameter for lambda functions needed in Smith G term
// This is a version for shape invariant (isotropic) NDFs
// Note: makse sure NdotS is not negative
BRDF_FUNCTION float Smith_G_a(float alpha, float NdotS)
{
	return NdotS / (fmaxf(0.00001f, alpha) * sqrtf(1.0f - fminf(0.99999f, NdotS * NdotS)));
}

// Lambda function for Smith G term derived for GGX distribution
BRDF_FUNCTION float Smith_G_Lambda_GGX(float a)
{
	return (-1.0f + sqrtf(1.0f + (1.0f / (a * a)))) * 0.5f;
}

// Lambda function for Smith G term derived for Beckmann distribution
// This is Walter's rational approximation (avoids evaluating of error function)
// Source: "Real-time Rendering", 4th edition, p.339 by Akenine-Moller et al.
// Note that this formulation is slightly optimized and different from Walter's
BRDF_FUNCTION float Smith_G_Lambda_Beckmann_Walter(float a)
{
	if (a < 1.6f) {
		return (1.0f - (1.259f - 0.396f * a) * a) / ((3.535f + 2.181f * a) * a);
		//return ((1.0f + (2.276f + 2.577f * a) * a) / ((3.535f + 2.181f * a) * a)) - 1.0f; //< Walter's original
	}
	else {
		return 0.0f;
	}
}

// Smith G1 term (masking function)
// This non-optimized version uses NDF specific lambda function (G_Lambda) resolved bia macro based on selected NDF
BRDF_FUNCTION float Smith_G1_General(float a)
{
	return 1.0f / (1.0f + Smith_G_Lambda(a));
}

// Smith G1 term (masking function) optimized for GGX distribution (by substituting G_Lambda_GGX into G1)
BRDF_FUNCTION float Smith_G1_GGX(float a)
{
	float a2 = a * a;
	return 2.0f / (sqrtf((a2 + 1.0f) / a2) + 1.0f);
}

// Smith G1 term (masking function) further optimized for GGX distribution (by substituting G_a into G1_GGX)
BRDF_FUNCTION float Smith_G1_GGX(float alpha, float NdotS, float alphaSquared, float NdotSSquared)
{
	return 2.0f / (sqrtf(((alphaSquared * (1.0f - NdotSSquared)) + NdotSSquared) / NdotSSquared) + 1.0f);
}

// Smith G1 term (masking function) optimized for Beckmann distribution (by substituting G_Lambda_Beckmann_Walter into G1)
// Source: "Microfacet Models for Refraction through Rough Surfaces" by Walter et al.
BRDF_FUNCTION float Smith_G1_Beckmann_Walter(float a)
{
	if (a < 1.6f) {
		return ((3.535f + 2.181f * a) * a) / (1.0f + (2.276f + 2.577f * a) * a);
	}
	else {
		return 1.0f;
	}
}

BRDF_FUNCTION float Smith_G1_Beckmann_Walter(float alpha, float NdotS, float alphaSquared, float NdotSSquared)
{
	return Smith_G1_Beckmann_Walter(Smith_G_a(alpha, NdotS));
}

// Smith G2 term (masking-shadowing function)
// Separable version assuming independent (uncorrelated) masking and shadowing, uses G1 functions for selected NDF
BRDF_FUNCTION float Smith_G2_Separable(float alpha, float NdotL, float NdotV)
{
	float aL = Smith_G_a(alpha, NdotL);
	float aV = Smith_G_a(alpha, NdotV);
	return Smith_G1(aL) * Smith_G1(aV);
}

// Smith G2 term (masking-shadowing function)
// Height correlated version - non-optimized, uses G_Lambda functions for selected NDF
BRDF_FUNCTION float Smith_G2_Height_Correlated(float alpha, float NdotL, float NdotV)
{
	float aL = Smith_G_a(alpha, NdotL);
	float aV = Smith_G_a(alpha, NdotV);
	return 1.0f / (1.0f + Smith_G_Lambda(aL) + Smith_G_Lambda(aV));
}

// Smith G2 term (masking-shadowing function) for GGX distribution
// Separable version assuming independent (uncorrelated) masking and shadowing - optimized by substituing G_Lambda for G_Lambda_GGX and 
// dividing by (4 * NdotL * NdotV) to cancel out these terms in specular BRDF denominator
// Source: "Moving Frostbite to Physically Based Rendering" by Lagarde & de Rousiers
// Note that returned value is G2 / (4 * NdotL * NdotV) and therefore includes division by specular BRDF denominator
BRDF_FUNCTION float Smith_G2_Separable_GGX_Lagarde(float alphaSquared, float NdotL, float NdotV)
{
	float a = NdotV + sqrtf(alphaSquared + NdotV * (NdotV - alphaSquared * NdotV));
	float b = NdotL + sqrtf(alphaSquared + NdotL * (NdotL - alphaSquared * NdotL));
	return 1.0f / (a * b);
}

// Smith G2 term (masking-shadowing function) for GGX distribution
// Height correlated version - optimized by substituing G_Lambda for G_Lambda_GGX and dividing by (4 * NdotL * NdotV) to cancel out 
// the terms in specular BRDF denominator
// Source: "Moving Frostbite to Physically Based Rendering" by Lagarde & de Rousiers
// Note that returned value is G2 / (4 * NdotL * NdotV) and therefore includes division by specular BRDF denominator
BRDF_FUNCTION float Smith_G2_Height_Correlated_GGX_Lagarde(float alphaSquared, float NdotL, float NdotV)
{
	float a = NdotV * sqrtf(alphaSquared + NdotL * (NdotL - alphaSquared * NdotL));
	float b = NdotL * sqrtf(alphaSquared + NdotV * (NdotV - alphaSquared * NdotV));
	return 0.5f / (a + b);
}

// Smith G2 term (masking-shadowing function) for GGX distribution
// Height correlated version - approximation by Hammon
// Source: "PBR Diffuse Lighting for GGX + Smith Microsurfaces", slide 84 by Hammon
// Note that returned value is G2 / (4 * NdotL * NdotV) and therefore includes division by specular BRDF denominator
BRDF_FUNCTION float Smith_G2_Height_Correlated_GGX_Hammon(float alpha, float NdotL, float NdotV)
{
	return 0.5f / (lerp(2.0f * NdotL * NdotV, NdotL + NdotV, alpha));
}

// A fraction G2/G1 where G2 is height correlated can be expressed using only G1 terms
// Source: "Implementing a Simple Anisotropic Rough Diffuse Material with Stochastic Evaluation", Appendix A by Heitz & Dupuy
BRDF_FUNCTION float Smith_G2_Over_G1_Height_Correlated(float alpha, float alphaSquared, float NdotL, float NdotV)
{
	float G1V = Smith_G1(alpha, NdotV, alphaSquared, NdotV * NdotV);
	float G1L = Smith_G1(alpha, NdotL, alphaSquared, NdotL * NdotL);
	return G1L / (G1V + G1L - G1V * G1L);
}

// Evaluates G2 for selected configuration (GGX/Beckmann, optimized/non-optimized, separable/height-correlated)
// Note that some paths aren't optimized too much...
// Also note that when USE_OPTIMIZED_G2 is specified, returned value will be: G2 / (4 * NdotL * NdotV) if GG-X is selected
BRDF_FUNCTION float Smith_G2(float alpha, float alphaSquared, float NdotL, float NdotV)
{
#if USE_OPTIMIZED_G2 && (MICROFACET_DISTRIBUTION == GGX)
#if USE_HEIGHT_CORRELATED_G2
#define G2_DIVIDED_BY_DENOMINATOR 1
	return Smith_G2_Height_Correlated_GGX_Lagarde(alphaSquared, NdotL, NdotV);
#else
#define G2_DIVIDED_BY_DENOMINATOR 1
	return Smith_G2_Separable_GGX_Lagarde(alphaSquared, NdotL, NdotV);
#endif
#else
#if USE_HEIGHT_CORRELATED_G2
	return Smith_G2_Height_Correlated(alpha, NdotL, NdotV);
#else
	return Smith_G2_Separable(alpha, NdotL, NdotV);
#endif
#endif
}

// -------------------------------------------------------------------------
//    Normal distribution functions
// -------------------------------------------------------------------------

BRDF_FUNCTION float Beckmann_D(float alphaSquared, float NdotH)
{
	float cos2Theta = NdotH * NdotH;
	float numerator = expf((cos2Theta - 1.0f) / (alphaSquared * cos2Theta));
	float denominator = PI_F * alphaSquared * cos2Theta * cos2Theta;
	return numerator / denominator;
}

BRDF_FUNCTION float GGX_D(float alphaSquared, float NdotH)
{
	float b = ((alphaSquared - 1.0f) * NdotH * NdotH + 1.0f);
	return alphaSquared / (PI_F * b * b);
}

// -------------------------------------------------------------------------
//    Microfacet model
// -------------------------------------------------------------------------

// Samples a microfacet normal for the GGX distribution using VNDF method.
// Source: "Sampling the GGX Distribution of Visible Normals" by Heitz
// Source: "Sampling Visible GGX Normals with Spherical Caps" by Dupuy & Benyoub
// Random variables 'u' must be in <0;1) interval
// PDF is 'G1(NdotV) * D'
BRDF_FUNCTION float3 sampleGGXVNDF(float3 Ve, float2 alpha2D, float2 u)
{
	// Section 3.2: transforming the view direction to the hemisphere configuration
	float3 Vh = normalize(make_float3(alpha2D.x * Ve.x, alpha2D.y * Ve.y, Ve.z));

#if USE_VNDF_WITH_SPHERICAL_CAPS

	// Source: "Sampling Visible GGX Normals with Spherical Caps" by Dupuy & Benyoub

	// Sample a spherical cap in (-Vh.z, 1]
	float phi = 2.0f * PI_F * u.x;
	float z = ((1.0f - u.y) * (1.0f + Vh.z)) - Vh.z;
	float sinTheta = sqrtf(saturatef(1.0f - z * z));
	float x = sinTheta * cosf(phi);
	float y = sinTheta * sinf(phi);

	// compute halfway direction;
	float3 Nh = make_float3(x, y, z) + Vh;

#else

	// Source: "Sampling the GGX Distribution of Visible Normals" by Heitz
	// See also https://hal.inria.fr/hal-00996995v1/document and http://jcgt.org/published/0007/04/01/

	// Section 4.1: orthonormal basis (with special case if cross product is zero)
	float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
	float3 T1 = lensq > 0.0f ? make_float3(-Vh.y, Vh.x, 0.0f) * rsqrtf(lensq) : make_float3(1.0f, 0.0f, 0.0f);
	float3 T2 = cross(Vh, T1);

	// Section 4.2: parameterization of the projected area
	float r = sqrtf(u.x);
	float phi = TWO_PI * u.y;
	float t1 = r * cosf(phi);
	float t2 = r * sinf(phi);
	float s = 0.5f * (1.0f + Vh.z);
	t2 = lerp(sqrtf(1.0f - t1 * t1), t2, s);

	// Section 4.3: reprojection onto hemisphere
	float3 Nh = t1 * T1 + t2 * T2 + sqrtf(fmaxf(0.0f, 1.0f - t1 * t1 - t2 * t2)) * Vh;

#endif

	// Section 3.4: transforming the normal back to the ellipsoid configuration
	return normalize(make_float3(alpha2D.x * Nh.x, alpha2D.y * Nh.y, fmaxf(0.0f, Nh.z)));
}

// PDF of sampling a reflection vector L using 'sampleGGXVNDF'.
// Note that PDF of sampling given microfacet normal is (G1 * D) when vectors are in local space (in the hemisphere around shading normal). 
// Remaining terms (1.0f / (4.0f * NdotV)) are specific for reflection case, and come from multiplying PDF by jacobian of reflection operator
BRDF_FUNCTION float sampleGGXVNDFReflectionPdf(float alpha, float alphaSquared, float NdotH, float NdotV, float LdotH)
{
	NdotH = fmaxf(0.00001f, NdotH);
	NdotV = fmaxf(0.00001f, NdotV);
	return (GGX_D(fmaxf(0.00001f, alphaSquared), NdotH) * Smith_G1_GGX(alpha, NdotV, alphaSquared, NdotV * NdotV)) / (4.0f * NdotV);
}

// "Walter's trick" is an adjustment of alpha value for Walter's sampling to reduce maximal weight of sample to about 4
// Source: "Microfacet Models for Refraction through Rough Surfaces" by Walter et al., page 8
BRDF_FUNCTION float waltersTrick(float alpha, float NdotV)
{
	return (1.2f - 0.2f * sqrtf(fabsf(NdotV))) * alpha;
}

// PDF of sampling a reflection vector L using 'sampleBeckmannWalter' or 'sampleGGXWalter'.
// Note that PDF of sampling given microfacet normal is (D * NdotH). Remaining terms (1.0f / (4.0f * LdotH)) are specific for
// reflection case, and come from multiplying PDF by jacobian of reflection operator
BRDF_FUNCTION float sampleWalterReflectionPdf(float alpha, float alphaSquared, float NdotH, float NdotV, float LdotH)
{
	NdotH = fmaxf(0.00001f, NdotH);
	LdotH = fmaxf(0.00001f, LdotH);
	return Microfacet_D(fmaxf(0.00001f, alphaSquared), NdotH) * NdotH / (4.0f * LdotH);
}

// Samples a microfacet normal for the Beckmann distribution using walter's method.
// Source: "Microfacet Models for Refraction through Rough Surfaces" by Walter et al.
// PDF is 'D * NdotH'
BRDF_FUNCTION float3 sampleBeckmannWalter(float3 Vlocal, float2 alpha2D, float2 u)
{
	float alpha = dot(alpha2D, make_float2(0.5f));

	// Equations (28) and (29) from Walter's paper for Beckmann distribution
	float tanThetaSquared = -(alpha * alpha) * logf(1.0f - u.x);
	float phi = TWO_PI_F * u.y;

	// Calculate cosTheta and sinTheta needed for conversion to H vector
	float cosTheta = rsqrtf(1.0f + tanThetaSquared);
	float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);

	// Convert sampled spherical coordinates to H vector
	return normalize(make_float3(sinTheta * cosf(phi), sinTheta * sinf(phi), cosTheta));
}

// Samples a microfacet normal for the GG-X distribution using walter's method.
// Source: "Microfacet Models for Refraction through Rough Surfaces" by Walter et al.
// PDF is 'D * NdotH'
BRDF_FUNCTION float3 sampleGGXWalter(float3 Vlocal, float2 alpha2D, float2 u)
{
	float alpha = dot(alpha2D, make_float2(0.5f));
	float alphaSquared = alpha * alpha;

	// Calculate cosTheta and sinTheta needed for conversion to H vector
	float cosThetaSquared = (1.0f - u.x) / ((alphaSquared - 1.0f) * u.x + 1.0f);
	float cosTheta = sqrtf(cosThetaSquared);
	float sinTheta = sqrtf(1.0f - cosThetaSquared);
	float phi = TWO_PI_F * u.y;

	// Convert sampled spherical coordinates to H vector
	return normalize(make_float3(sinTheta * cosf(phi), sinTheta * sinf(phi), cosTheta));
}

// Weight for the reflection ray sampled from GGX distribution using VNDF method
BRDF_FUNCTION float specularSampleWeightGGXVNDF(float alpha, float alphaSquared, float NdotL, float NdotV, float HdotL, float NdotH)
{
#if USE_HEIGHT_CORRELATED_G2
	return Smith_G2_Over_G1_Height_Correlated(alpha, alphaSquared, NdotL, NdotV);
#else 
	return Smith_G1_GGX(alpha, NdotL, alphaSquared, NdotL * NdotL);
#endif
}

// Weight for the reflection ray sampled from Beckmann distribution using Walter's method
BRDF_FUNCTION float specularSampleWeightBeckmannWalter(float alpha, float alphaSquared, float NdotL, float NdotV, float HdotL, float NdotH)
{
	return (HdotL * Smith_G2(alpha, alphaSquared, NdotL, NdotV)) / (NdotV * NdotH);
}

// Weight for the reflection ray sampled from GGX distribution using Walter's method
BRDF_FUNCTION float specularSampleWeightGGXWalter(float alpha, float alphaSquared, float NdotL, float NdotV, float HdotL, float NdotH)
{
#if USE_OPTIMIZED_G2 
	return (NdotL * HdotL * Smith_G2(alpha, alphaSquared, NdotL, NdotV) * 4.0f) / NdotH;
#else
	return (HdotL * Smith_G2(alpha, alphaSquared, NdotL, NdotV)) / (NdotV * NdotH);
#endif
}

// Samples a reflection ray from the rough surface using selected microfacet distribution and sampling method
// Resulting weight includes multiplication by cosine (NdotL) term
BRDF_FUNCTION float3 sampleSpecularMicrofacet(float3 Vlocal, float alpha, float alphaSquared, float3 specularF0, float2 u, float3& weight)
{
	// Sample a microfacet normal (H) in local space
	float3 Hlocal;
	if (alpha == 0.0f) {
		// Fast path for zero roughness (perfect reflection), also prevents NaNs appearing due to divisions by zeroes
		Hlocal = float3_up;
	}
	else {
		// For non-zero roughness, this calls VNDF sampling for GG-X distribution or Walter's sampling for Beckmann distribution
		Hlocal = sampleSpecularHalfVector(Vlocal, make_float2(alpha), u);
	}

	// Reflect view direction to obtain light vector
	float3 Llocal = reflect(-Vlocal, Hlocal);

	// Note: HdotL is same as HdotV here
	// Clamp dot products here to small value to prevent numerical instability. Assume that rays incident from below the hemisphere have been filtered
	float HdotL = fmaxf(0.00001f, fminf(1.0f, dot(Hlocal, Llocal)));
	const float3 Nlocal = float3_up;
	float NdotL = fmaxf(0.00001f, fminf(1.0f, dot(Nlocal, Llocal)));
	float NdotV = fmaxf(0.00001f, fminf(1.0f, dot(Nlocal, Vlocal)));
	float NdotH = fmaxf(0.00001f, fminf(1.0f, dot(Nlocal, Hlocal)));
	float3 F = evalFresnel(specularF0, shadowedF90(specularF0), HdotL);

	// Calculate weight of the sample specific for selected sampling method 
	// (this is microfacet BRDF divided by PDF of sampling method - notice how most terms cancel out)
	weight = F * specularSampleWeight(alpha, alphaSquared, NdotL, NdotV, HdotL, NdotH);

	return Llocal;
}

// Evaluates microfacet specular BRDF
BRDF_FUNCTION float3 evalMicrofacet(const BrdfData data)
{
	float D = Microfacet_D(fmaxf(0.00001f, data.alphaSquared), data.NdotH);
	float G2 = Smith_G2(data.alpha, data.alphaSquared, data.NdotL, data.NdotV);
	//float3 F = evalFresnel(data.specularF0, shadowedF90(data.specularF0), data.VdotH); //< Unused, F is precomputed already

#if G2_DIVIDED_BY_DENOMINATOR
	return data.F * (G2 * D * data.NdotL);
#else
	return ((data.F * G2 * D) / (4.0f * data.NdotL * data.NdotV)) * data.NdotL;
#endif
}

// -------------------------------------------------------------------------
//    Combined BRDF
// -------------------------------------------------------------------------

// Precalculates commonly used terms in BRDF evaluation
// Clamps around dot products prevent NaNs and ensure numerical stability, but make sure to
// correctly ignore rays outside of the sampling hemisphere, by using 'Vbackfacing' and 'Lbackfacing' flags
BRDF_FUNCTION BrdfData prepareBRDFData(float3 N, float3 L, float3 V, MaterialProperties material)
{
	BrdfData data;

	// Evaluate VNHL vectors
	data.V = V;
	data.N = N;
	data.H = normalize(L + V);
	data.L = L;

	float NdotL = dot(N, L);
	float NdotV = dot(N, V);
	data.Vbackfacing = (NdotV <= 0.0f);
	data.Lbackfacing = (NdotL <= 0.0f);

	// Clamp NdotS to prevent numerical instability. Assume vectors below the hemisphere will be filtered using 'Vbackfacing' and 'Lbackfacing' flags
	data.NdotL = fminf(fmaxf(0.00001f, NdotL), 1.0f);
	data.NdotV = fminf(fmaxf(0.00001f, NdotV), 1.0f);

	data.LdotH = saturatef(dot(L, data.H));
	data.NdotH = saturatef(dot(N, data.H));
	data.VdotH = saturatef(dot(V, data.H));

	// Unpack material properties
	data.specularF0 = baseColorToSpecularF0(material.baseColor, material.metalness, material.reflectance);
	data.diffuseReflectance = baseColorToDiffuseReflectance(material.baseColor, material.metalness);

	// Unpack 'perceptively linear' -> 'linear' -> 'squared' roughness
	data.roughness = material.roughness;
	data.alpha = material.roughness * material.roughness;
	data.alphaSquared = data.alpha * data.alpha;

	// Pre-calculate some more BRDF terms
	data.F = evalFresnel(data.specularF0, shadowedF90(data.specularF0), data.LdotH);

	return data;
}

// This is an entry point for evaluation of all other BRDFs based on selected configuration (for direct light)
BRDF_FUNCTION float3 evalCombinedBRDF(float& pdf, float3 N, float3 L, float3 V, MaterialProperties material)
{
	// Prepare data needed for BRDF evaluation - unpack material properties and evaluate commonly used terms (e.g. Fresnel, NdotL, ...)
	const BrdfData data = prepareBRDFData(N, L, V, material);

	// Ignore V and L rays "below" the hemisphere
	if (data.Vbackfacing || data.Lbackfacing) return float3_zero;

	// Eval specular and diffuse BRDFs
	float3 specular = evalSpecular(data);
	float3 diffuse = evalDiffuse(data);

	float diffuse_pdf = diffusePdf(data.NdotL);
	float specular_pdf = specularPdf(data.alpha, data.alphaSquared, data.NdotH, data.NdotV, data.LdotH);

	// Combine specular and diffuse layers
#if COMBINE_BRDFS_WITH_FRESNEL
	// Specular is already multiplied by F, just attenuate diffuse
	pdf = (1.0f - luminance(data.F)) * diffuse_pdf + specular_pdf;
	return (float3_one - data.F) * diffuse + specular;
#else
	pdf = diffuse_pdf + specular_pdf;
	return diffuse + specular;
#endif
}

// This is an entry point for evaluation of all other BRDFs based on selected configuration (for indirect light)
BRDF_FUNCTION bool evalIndirectCombinedBRDF(float3& rayDirection, float3& sampleWeight, float3 shadingNormal, float3 geometryNormal, float3 V, MaterialProperties material, float2 u, const bool isSpecular)
{
	// Ignore incident ray coming from "below" the hemisphere
	if (dot(shadingNormal, V) <= 0.0f) return false;

	// Transform view direction into local space of our sampling routines 
	// (local space is oriented so that its positive Z axis points along the shading normal)
	float4 qRotationToZ = getRotationToZAxis(shadingNormal);
	float3 Vlocal = rotatePoint(qRotationToZ, V);
	/*
	float worldToLocal[3][3];
	float localToWorld[3][3];
	createWorldToLocalMatrix(shadingNormal, worldToLocal);
	invertMatrix(worldToLocal, localToWorld);
	float3 Vlocal = normalize(matrixMultiply(V, worldToLocal));
	*/
	const float3 Nlocal = float3_up;

	float3 rayDirectionLocal = float3_zero;

	if (!isSpecular) {

		// Sample diffuse ray using cosine-weighted hemisphere sampling 
		rayDirectionLocal = sampleHemisphere(u);
		const BrdfData data = prepareBRDFData(Nlocal, rayDirectionLocal, Vlocal, material);

		// Function 'diffuseTerm' is predivided by PDF of sampling the cosine weighted hemisphere
		sampleWeight = data.diffuseReflectance * diffuseTerm(data);

#if COMBINE_BRDFS_WITH_FRESNEL		
		// Sample a half-vector of specular BRDF. Note that we're reusing random variable 'u' here, but correctly it should be an new independent random number
		float3 Hspecular = sampleSpecularHalfVector(Vlocal, make_float2(data.alpha), u);

#if USE_WALTER_GGX_SAMPLING
		// Check if specular sample is valid (does not reflect under the hemisphere)
		float VdotH = dot(Vlocal, Hspecular);
		if (VdotH > 0.00001f)
		{
			sampleWeight *= (float3_one - evalFresnel(data.specularF0, shadowedF90(data.specularF0), fminf(1.0f, VdotH)));
		}
		else {
			return false;
		}
#else
		// Clamp HdotL to small value to prevent numerical instability. Assume that rays incident from below the hemisphere have been filtered
		// Note: VdotH is always positive for VNDF sampling so we don't need to test if it's positive like we do for sampling with Walter's method
		float VdotH = fmaxf(0.00001f, fminf(1.0f, dot(Vlocal, Hspecular)));
		sampleWeight *= (float3_one - evalFresnel(data.specularF0, shadowedF90(data.specularF0), VdotH));
#endif
#endif

	}
	else if (isSpecular) {
		const BrdfData data = prepareBRDFData(Nlocal, float3_up /* unused L vector */, Vlocal, material);
		rayDirectionLocal = sampleSpecular(Vlocal, data.alpha, data.alphaSquared, data.specularF0, u, sampleWeight);
	}

	// Prevent tracing direction with no contribution
	if (luminance(sampleWeight) == 0.0f) return false;

	// Transform sampled direction Llocal back to V vector space
	rayDirection = normalize(rotatePoint(invertRotation(qRotationToZ), rayDirectionLocal));
	//ayDirection = normalize(matrixMultiply(rayDirectionLocal, localToWorld));

	// Prevent tracing direction "under" the hemisphere (behind the triangle)
	if (dot(geometryNormal, rayDirection) <= 0.0f) return false;

	return true;
}

// Calculates probability of selecting BRDF (specular or diffuse) using the approximate Fresnel term
BRDF_FUNCTION float getBrdfProbability(float3 albedo, float metalness, float3 V, float3 shadingNormal)
{
	// Evaluate Fresnel term using the shading normal
	// Note: we use the shading normal instead of the microfacet normal (half-vector) for Fresnel term here. That's suboptimal for rough surfaces at grazing angles, but half-vector is yet unknown at this point
	float specularF0 = luminance(baseColorToSpecularF0(albedo, metalness));
	float diffuseReflectance = luminance(baseColorToDiffuseReflectance(albedo, metalness));
	float Fresnel = saturatef(luminance(evalFresnel(make_float3(specularF0, specularF0, specularF0), shadowedF90(make_float3(specularF0, specularF0, specularF0)), fmaxf(0.0f, dot(V, shadingNormal)))));

	// Approximate relative contribution of BRDFs using the Fresnel term
	float specular = Fresnel;
	float diffuse = diffuseReflectance * (1.0f - Fresnel); //< If diffuse term is weighted by Fresnel, apply it here as well

	// Return probability of selecting specular BRDF over diffuse BRDF
	float p = (specular / fmaxf(0.0001f, (specular + diffuse)));

	// Clamp probability to avoid undersampling of less prominent BRDF
	return fminf(fmaxf(p, 0.0f), 1.0f);
}

} // namespace brdf