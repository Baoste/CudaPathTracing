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
    __device__ inline double3 getFr(const Ray& ray, const double3 direction, const double3 wm = make_double3(0.0,0.0,0.0))
    {
        return (*material).fr(ray, normal, direction, wm, frontFace);
    }
    __device__ inline void getSample(const Ray& ray, double3& direction, double3& wm, double& pdf, double r1, double r2, double r3)
    {
        (*material).sampleLight(ray, normal, direction, wm, pdf, r1, r2, r3, frontFace);
    }
};