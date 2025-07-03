#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "device_launch_parameters.h"
#include <cmath>

#define INF 999999.0
#define PI 3.14159265358979323846

#define mMin(a, b) ((a) < (b) ? (a) : (b))
#define mMax(a, b) ((a) > (b) ? (a) : (b))
#define mSwap(a, b) { double temp = a; a = b; b = temp; }

enum MaterialType { M_LIGHT, M_OPAQUE, M_SPECULAR_DIELECTRIC, M_ROUGH_DIELECTRIC };

__host__ __device__ inline double3 operator+(const double3& v1, const double3& v2) 
{
    return make_double3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}

__host__ __device__ inline double3 operator-(const double3& v1, const double3& v2)
{
    return make_double3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}

__host__ __device__ inline double3 operator-(const double3& v)
{
    return make_double3(-v.x, -v.y, -v.z);
}

__host__ __device__ inline double3 operator*(const double3& v1, const double3& v2)
{
    return make_double3(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);
}

__host__ __device__ inline double3 operator*(double t, const double3& v)
{
    return make_double3(v.x * t, v.y * t, v.z * t);
}

__host__ __device__ inline double3 operator*(const double3& v, double t)
{
    return make_double3(v.x * t, v.y * t, v.z * t);
}

__host__ __device__ inline double3 operator/(double3 v, double t)
{
    return make_double3(v.x / t, v.y / t, v.z / t);
}

__host__ __device__ inline void operator+=(double3& v1, const double3& v2)
{
    v1.x += v2.x;
    v1.y += v2.y;
    v1.z += v2.z;
}

__host__ __device__ inline void operator-=(double3& v1, const double3& v2)
{
    v1.x -= v2.x;
    v1.y -= v2.y;
    v1.z -= v2.z;
}

__host__ __device__ inline void operator*=(double3& v, const double& t)
{
    v.x *= t;
    v.y *= t;
    v.z *= t;
}

__host__ __device__ inline void operator*=(double3& v1, const double3& v2)
{
    v1.x *= v2.x;
    v1.y *= v2.y;
    v1.z *= v2.z;
}

__host__ __device__ inline void operator/=(double3& v, const double& t)
{
    v.x /= t;
    v.y /= t;
    v.z /= t;
}

__host__ __device__ inline uchar4 operator+(const uchar4& v1, const uchar4& v2)
{
    return make_uchar4(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z, v1.w + v2.w);
}

__host__ __device__ inline uchar4 operator/(const uchar4& v1, const unsigned char t)
{
    return make_uchar4(v1.x / t, v1.y / t, v1.z / t, v1.w / t);
}

__host__ __device__ inline double SquaredLength(const double3& v)
{
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

__host__ __device__ inline double Length(const double3& v)
{
    return std::sqrt(SquaredLength(v));
}

__host__ __device__ inline double Dot(const double3& v1, const double3& v2)
{
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

__host__ __device__ inline double3 Cross(const double3& v1, const double3& v2)
{
    return make_double3(v1.y * v2.z - v1.z * v2.y,
        v1.z * v2.x - v1.x * v2.z,
        v1.x * v2.y - v1.y * v2.x);
}

__host__ __device__ inline double3 Unit(const double3& v)
{
    return v / Length(v);
}

using Color = double3;

// Complex
using Complex = double2;
__host__ __device__ inline Complex operator/(const Complex& v1, const Complex& v2)
{
    double a = v1.x, b = v1.y;
    double c = v2.x, d = v2.y;

    double denominator = c * c + d * d;
    return make_double2(
        (a * c + b * d) / denominator,
        (b * c - a * d) / denominator
    );
}

__host__ __device__ inline Complex operator*(double t, const Complex& v)
{
    return make_double2(t * v.x, t * v.y);
}

__host__ __device__ inline Complex operator*(const Complex& v, double t)
{
    return make_double2(t * v.x, t * v.y);
}

__host__ __device__ inline Complex operator+(double t, const Complex& v)
{
    return make_double2(t + v.x, v.y);
}

__host__ __device__ inline Complex operator+(const Complex& v, double t)
{
    return make_double2(t + v.x, v.y);
}

__host__ __device__ inline Complex operator-(double t, const Complex& v)
{
    return make_double2(t - v.x, -v.y);
}

__host__ __device__ inline Complex operator-(const Complex& v, double t)
{
    return make_double2(v.x - t, v.y);
}

__host__ __device__ inline double Norm(const Complex& v)
{
    return v.x * v.x + v.y * v.y;
}

// Degree
__host__ __device__ inline double DegreesToRadians(double degrees)
{
    return degrees * PI / 180.0;
}