#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "device_launch_parameters.h"
#include <cmath>

#define INF 999999
#define PI 3.14159265358979323846

__host__ __device__ inline double3 operator+(const double3& v1, const double3& v2) 
{
    return make_double3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}
#include "rtmath.cuh"

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

__host__ __device__ inline double DegreesToRadians(double degrees)
{
    return degrees * PI / 180.0;
}