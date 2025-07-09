#pragma once

#include "Material.cuh"

class HitRecord
{
public:
    double3 hitPos;
    double3 hitColor;
    double3 hitWm;
    double3 normal;  //! outwardNoraml
    Material* material;
    double t;
    bool frontFace;

public:
    __device__ HitRecord()
        : hitPos(make_double3(0.0, 0.0, 0.0)),
        hitColor(make_double3(-1.0, -1.0, -1.0)),
        hitWm(make_double3(0.0, 0.0, 0.0)),
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

    __device__ void sampleTexture(const double3 color)
    {
        hitColor = color;
    }
    __device__ void sampleTexture(const double3 color, const unsigned char* texture, const double width, const double height, const double uu, const double vv)
    {
        if (texture == NULL)
        {
            sampleTexture(color);
            return;
        }
        int x = mMin(mMax(int(uu * width), 0), width - 1);
        int y = mMin(mMax(int(vv * height), 0), height - 1);
        int idx = (y * width + x) * 3;
        hitColor = make_double3(texture[idx] / 255.0, texture[idx + 1] / 255.0, texture[idx + 2] / 255.0);
    }
    __device__ inline double3 getFr(const Ray& ray, const double3 direction)
    {
        return (*material).fr(hitColor, ray, normal, direction, hitWm, frontFace);
    }
    __device__ inline void getSample(const Ray& ray, double3& direction, double& pdf, double r1, double r2, double r3)
    {
        (*material).sampleLight(ray, normal, direction, hitWm, pdf, r1, r2, r3, frontFace);
    }
};