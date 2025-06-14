#pragma once

#include "HitRecord.cuh"

enum ObjectType { NONE, SPHERE };

class Sphere
{
private:
    double3 center;
    double radius;

public:
    __host__ __device__ Sphere()
        : center(make_double3(0.0, 0.0, 0.0)), radius(1.0)
    {
    }
    __host__ __device__ Sphere(const double3& _center, double _radius)
        : center(_center), radius(_radius) 
    {
    }
    __host__ __device__ ~Sphere() {}
    __device__ inline bool hit(const Ray& ray, HitRecord& record, double t_min, double t_max) const
    {
        double a = Dot(ray.direction, ray.direction);
        double b = Dot(ray.direction, ray.origin - center);
        double c = Dot(ray.origin - center, ray.origin - center) - radius * radius;
        double delta = b * b - a * c;
        // if no hit, return false
        if (delta < 0)
            return false;
        // if hit. caculate t, from light = o+td
        double t = (-b - sqrt(delta)) / a;
        // if t is out of range, try another t
        if (t < t_min || t > t_max)
            t = (-b + sqrt(delta)) / a;
        // if still out of range, return false
        if (t < t_min || t > t_max)
            return false;

        record.hitPos = ray.at(t);
        record.t = t;
        double3 outwardNoraml = (record.hitPos - center) / radius;
        record.setFaceNormal(ray, outwardNoraml);

        return true;
    }
};

class Hittable
{
public:
    ObjectType type;
    union {
        Sphere sphere;
    };

public:
    __host__ __device__ Hittable() : type(ObjectType::NONE) {}
    __host__ __device__ Hittable(const Sphere& s) : type(ObjectType::SPHERE), sphere(s) {}
    __host__ __device__ ~Hittable() {}
    __device__ inline bool hit(const Ray& ray, HitRecord& record, double t_min, double t_max) const
    {
        switch (type)
        {
        case ObjectType::SPHERE:
            return sphere.hit(ray, record, t_min, t_max);
        case ObjectType::NONE:
        default:
            return false;
        }
    }
};