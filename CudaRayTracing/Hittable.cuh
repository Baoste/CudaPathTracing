#pragma once

#include "HitRecord.cuh"
#include "AABB.cuh"
#include "Material.cuh"


enum ObjectType { NONE, SPHERE, LIGHT, TRIANGLE };

class Sphere
{
public:
    double3 center;
    double radius;
    Material material;

public:
    __host__ __device__ Sphere()
        : center(make_double3(0.0, 0.0, 0.0)), radius(1.0), material(make_double3(1.0, 1.0, 1.0))
    {
    }
    __host__ __device__ Sphere(const double3 _center, double _radius, double3 color, bool glass = false)
        : center(_center), radius(_radius), material(color, glass)
    {
    }
    __host__ __device__ ~Sphere() {}
    __device__ inline bool hit(const Ray& ray, HitRecord& record, double t_min, double t_max)
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
        record.material = &material;
        double3 outwardNoraml = (record.hitPos - center) / radius;
        record.setFaceNormal(ray, outwardNoraml);

        return true;
    }
};

class Triangle
{
public:
    double3 p0, p1, p2;
    double3 center;
    double3 normal;
    Material material;

public:
    __host__ __device__ Triangle(const double3 _p0, const double3 _p1, const double3 _p2, const double3 color, bool glass = false)
        : p0(_p0), p1(_p1), p2(_p2), material(color, glass)
    {
        center = (p0 + p1 + p2) / 3.0;
        normal = Unit(Cross(p1 - p0, p2 - p0));
    }
    __host__ __device__ ~Triangle() {}
    __device__ inline bool hit(const Ray& ray, HitRecord& record, double t_min, double t_max)
    {
        const double EPSILON = 1e-8;

        double3 edge1 = p1 - p0;
        double3 edge2 = p2 - p0;

        double3 h = Cross(ray.direction, edge2);
        double a = Dot(edge1, h);
        // parrallel
        if (fabs(a) < EPSILON)
            return false;

        double f = 1.0 / a;
        double3 s = ray.origin - p0;
        double u = f * Dot(s, h);
        if (u < 0.0 || u > 1.0)
            return false;

        double3 q = Cross(s, edge1);
        double v = f * Dot(ray.direction, q);
        if (v < 0.0 || u + v > 1.0)
            return false;

        double t = f * Dot(edge2, q);
        if (t < t_min || t > t_max)
            return false;

        record.hitPos = ray.at(t);
        record.t = t;
        record.material = &material;
        double3 outwardNoraml = Unit(Cross(edge1, edge2));
        record.setFaceNormal(ray, outwardNoraml);

        return true;
    }
};

class Light
{
public:
    double3 center;
    double width;
    double height;
    double3 normal;
    double3 edgeU;
    double3 edgeV;
    Color color;
    bool visible;

public:
    __host__ __device__ Light(double3 _center, double _width, double _height, double3 _normal, double3 _color, bool _visible = false)
        : center(_center), width(_width), height(_height), normal(_normal), color(_color), visible(_visible)
    {
        normal = Unit(normal);
        edgeU = fabs(normal.x) > 0.9 ? Unit(Cross(make_double3(0, 1, 0), normal)) : Unit(Cross(make_double3(1, 0, 0), normal));
        edgeV = Unit(Cross(normal, edgeU));
    }
    __host__ __device__ ~Light() {}
    __device__ inline bool hit(const Ray& ray, HitRecord& record, double t_min, double t_max)
    {
        if (!visible)
            return false;

        double denom = Dot(ray.direction, normal);
        // parallel or back face
        if (fabs(denom) < 1e-6)
            return false;
        double t = Dot(center - ray.origin, normal) / denom;
        // back face or far away
        if (t < 0)
            return false;

        double3 hitPoint = ray.at(t);
        double3 diff = hitPoint - center;
        double u = Dot(diff, edgeU);
        double v = Dot(diff, edgeV);
        if (fabs(u) <= width / 2.0 && fabs(v) <= height / 2.0)
        {
            record.hitPos = hitPoint;
            record.t = t;
            record.material = NULL;
            double3 outwardNoraml = normal;
            record.setFaceNormal(ray, outwardNoraml);
            return true;
        }
        return false;
    }
};


class Hittable
{
public:
    ObjectType type;
    AABB aabb;
    double3 center;
    union {
        Sphere sphere;
        Light light;
        Triangle triangle;
    };

public:
    __host__ __device__ Hittable() : center(make_double3(0.0, 0.0, 0.0)), type(ObjectType::NONE) {}
    // Sphere constructor
    __host__ __device__ Hittable(const Sphere& s)
        : center(make_double3(s.center.x, s.center.y, s.center.z)), type(ObjectType::SPHERE), sphere(s) 
    {
        aabb = AABB(s.center - make_double3(s.radius, s.radius, s.radius), s.center + make_double3(s.radius, s.radius, s.radius));
    }
    // Light constructor
    __host__ __device__ Hittable(const Light& l)
        : center(make_double3(l.center.x, l.center.y, l.center.z)), type(ObjectType::LIGHT), light(l)
    {
        double3 N = Unit(l.normal);
        double3 T = fabs(N.x) > 0.9 ?
            Unit(Cross(make_double3(0, 1, 0), N)) :
            Unit(Cross(make_double3(1, 0, 0), N));
        double3 B = Unit(Cross(N, T));
        // 构造四个角点（也可以8个点扩展成体积）
        double3 corners[4] = {
            l.center + T * (l.width / 2) + B * (l.height / 2),
            l.center + T * (l.width / 2) + B * (-l.height / 2),
            l.center + T * (-l.width / 2) + B * (l.height / 2),
            l.center + T * (-l.width / 2) + B * (-l.height / 2)
        };

        // 构造 AABB（点集取 min/max）
        double3 aabbMin = corners[0];
        double3 aabbMax = corners[0];
        for (int i = 1; i < 4; ++i) 
        {
            aabbMin.x = mMin(aabbMin.x, corners[i].x);
            aabbMin.y = mMin(aabbMin.y, corners[i].y);
            aabbMin.z = mMin(aabbMin.z, corners[i].z);
            aabbMax.x = mMax(aabbMax.x, corners[i].x);
            aabbMax.y = mMax(aabbMax.y, corners[i].y);
            aabbMax.z = mMax(aabbMax.z, corners[i].z);
        }

        aabb = AABB(aabbMin, aabbMax).pad();
    }
    // Triangle constructor
    __host__ __device__ Hittable(const Triangle& t)
        : type(ObjectType::TRIANGLE), triangle(t)
    {
        center = (t.p0 + t.p1 + t.p2) / 3.0;
        aabb = AABB(make_double3(mMin(mMin(t.p0.x, t.p1.x), t.p2.x), mMin(mMin(t.p0.y, t.p1.y), t.p2.y), mMin(mMin(t.p0.z, t.p1.z), t.p2.z)),
                    make_double3(mMax(mMax(t.p0.x, t.p1.x), t.p2.x), mMax(mMax(t.p0.y, t.p1.y), t.p2.y), mMax(mMax(t.p0.z, t.p1.z), t.p2.z))).pad();
    }

    __host__ __device__ ~Hittable() {}
    __device__ inline bool hit(const Ray& ray, HitRecord& record, double t_min, double t_max)
    {
        switch (type)
        {
        case ObjectType::SPHERE:
            return sphere.hit(ray, record, t_min, t_max);
        case ObjectType::LIGHT:
            return light.hit(ray, record, t_min, t_max);
        case ObjectType::TRIANGLE:
            return triangle.hit(ray, record, t_min, t_max);
        case ObjectType::NONE:
        default:
            return false;
        }
    }
};