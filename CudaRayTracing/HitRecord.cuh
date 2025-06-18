#pragma once

#include "Material.cuh"

class HitRecord
{
public:
    double3 hitPos;
    double3 normal;  //! outwardNoraml
    Material* material;
    double t;
    bool frontFace;

public:
    __device__ HitRecord()
        : hitPos(make_double3(0.0, 0.0, 0.0)),
        normal(make_double3(0.0, 0.0, 0.0)),
        material(NULL),
        t(INF),
        frontFace(false) 
    {
    }
    __device__ inline void setFaceNormal(const Ray& ray, const double3 outwardNormal) 
    {
        frontFace = Dot(ray.direction, outwardNormal) < 0;
        normal = frontFace ? outwardNormal : -outwardNormal;
    }
    __device__ inline double3 getFr(const Ray& ray, const double3 direction)
    {
        return (*material).fr(ray, normal, direction);
    }
    __device__ inline void getSample(const Ray& ray, double3& direction, double& pdf, double r1, double r2)
    {
        (*material).sampleLight(ray, normal, direction, pdf, r1, r2, frontFace);
    }
};