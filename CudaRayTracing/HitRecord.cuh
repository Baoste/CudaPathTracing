#pragma once

#include "Ray.cuh"

class HitRecord
{
public:
    double3 hitPos;
    double3 normal;  //! outwardNoraml
    double t;
    bool frontFace;

public:
    __device__ HitRecord()
        : hitPos(make_double3(0.0, 0.0, 0.0)),
        normal(make_double3(0.0, 0.0, 0.0)),
        t(INF),
        frontFace(false) 
    {
    }
    __device__ inline void setFaceNormal(const Ray& ray, const double3& outwardNormal) 
    {
        frontFace = Dot(ray.direction, outwardNormal) < 0;
        normal = frontFace ? outwardNormal : -outwardNormal;
    }
};