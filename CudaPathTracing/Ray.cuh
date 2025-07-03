#pragma once

#include "rtmath.cuh"

class Ray
{
public:
    __device__ Ray()
        : origin(make_double3(0.0, 0.0, 0.0)),
        direction(make_double3(0.0, 0.0, 0.0)),
        time(0.0) {
    }
    __device__ Ray(const double3 o, const double3 d, double t = 0.0)
        : origin(make_double3(o.x, o.y, o.z)),
        direction(make_double3(d.x, d.y, d.z)),
        time(t) {
    }
    __device__ inline double3 at(const double& t) const { return origin + t * direction; }

public:
    double3 origin;
    double3 direction;
    double time;
};