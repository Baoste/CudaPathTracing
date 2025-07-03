#pragma once

#include "HitRecord.cuh"
#include "AABB.cuh"
#include "Material.cuh"


enum ObjectType { NONE, SPHERE, LIGHT, TRIANGLE, BEZIER };

class Sphere
{
public:
    double3 center;
    double radius;
    Material material;

public:
    __host__ __device__ Sphere()
        : center(make_double3(0.0, 0.0, 0.0)), radius(1.0), material(make_double3(1.0, 1.0, 1.0), 0.5, 0.5, MaterialType::M_OPAQUE)
    {
    }
    __host__ __device__ Sphere(const double3 _center, double _radius, double3 color, double alphaX, double alphaY, MaterialType type = MaterialType::M_OPAQUE)
        : center(_center), radius(_radius), material(color, alphaX, alphaY, type)
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
        if (delta < 0.0)
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
    double2 uv0, uv1, uv2;
    double3 center;
    double3 normal;
    Material material;
    unsigned char* texture;
    int width, height, channels;

public:
    __host__ __device__ Triangle(const double3 _p0, const double3 _p1, const double3 _p2, 
        const double3 color, double alphaX, double alphaY, MaterialType type = MaterialType::M_OPAQUE,
        const double2 _uv0 = make_double2(0.0, 0.0), const double2 _uv1 = make_double2(0.0, 0.0), const double2 _uv2 = make_double2(0.0, 0.0),
        unsigned char* _texture = NULL, int _width = 0, int _height = 0, int _channels = 0)
        : p0(_p0), p1(_p1), p2(_p2), uv0(_uv0), uv1(_uv1), uv2(_uv2), material(color, alphaX, alphaY, type),
          texture(_texture), width(_width), height(_height), channels(_channels)
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

        // get texture
        double uu = (1.0 - u - v) * uv0.x + u * uv1.x + v * uv2.x;
        double vv = (1.0 - u - v) * uv0.y + u * uv1.y + v * uv2.y;
        record.sampleTexture(texture, width, height, uu, vv);

        record.hitPos = ray.at(t);
        record.t = t;
        record.material = &material;
        double3 outwardNoraml = Unit(Cross(edge1, edge2));
        record.setFaceNormal(ray, outwardNoraml);

        return true;
    }
};

//class Bzeier
//{
//public:
//    double3 p[4][4];
//    double3 center;
//    double3 normal;
//    Material material;
//public:
//    __host__ __device__ Bzeier(const double3 _p0, const double3 _p1, const double3 _p2, const double3 _p3,
//        const double3 _p4, const double3 _p5, const double3 _p6, const double3 _p7,
//        const double3 _p8, const double3 _p9, const double3 _p10, const double3 _p11,
//        const double3 _p12, const double3 _p13, const double3 _p14, const double3 _p15,
//        const double3 color, double alphaX, double alphaY, bool glass = false)
//        : material(color, alphaX, alphaY, glass)
//    {
//        p[0][0] = _p0;
//        p[0][1] = _p1;
//        p[0][2] = _p2;
//        p[0][3] = _p3;
//
//        p[1][0] = _p4;
//        p[1][1] = _p5;
//        p[1][2] = _p6;
//        p[1][3] = _p7;
//
//        p[2][0] = _p8;
//        p[2][1] = _p9;
//        p[2][2] = _p10;
//        p[2][3] = _p11;
//
//        p[3][0] = _p12;
//        p[3][1] = _p13;
//        p[3][2] = _p14;
//        p[3][3] = _p15;
//
//        center = (_p0 + _p3 + _p12 + _p15) / 4.0;
//        normal = Unit(Cross(_p1 - _p0, _p4 - _p0));
//    }
//    __host__ __device__ Bzeier(const Bzeier& b)
//        : material(b.material.color, b.material.alphaX, b.material.alphaY, b.material.glass)
//    {
//        for (int i = 0; i < 4; ++i)
//            for (int j = 0; j < 4; ++j)
//                p[i][j] = b.p[i][j];
//
//        center = b.center;
//        normal = b.normal;
//    }
//    __host__ __device__ ~Bzeier() {}
//    __device__ inline bool hit(const Ray& ray, HitRecord& record, double t_min, double t_max)
//    {
//        double2 uv = isPointOnBezier(ray, record, t_min, t_max);
//        if (uv.x < 0.0 || uv.y < 0.0)
//            return false;
//
//        double3 derU = make_double3(0.0, 0.0, 0.0);
//        double3 derV = make_double3(0.0, 0.0, 0.0);
//
//        for (int i = 0; i < 3; i++)
//        {
//            for (int j = 0; j < 4; j++)
//            {
//                derU += 3.0 * bernstein2(i, uv.x) * bernstein3(j, uv.y) * (p[i + 1][j] - p[i][j]);
//            }
//        }
//        for (int i = 0; i < 4; i++)
//        {
//            for (int j = 0; j < 3; j++)
//            {
//                derV += 3 * bernstein3(i, uv.x) * bernstein2(j, uv.y) * (p[i][j + 1] - p[i][j]);
//            }
//        }
//        
//        double3 outwardNoraml = Unit(Cross(derU, derV));
//        record.setFaceNormal(ray, outwardNoraml);
//
//        return true;
//    }
//private:
//    __device__ inline double bernstein3(int i, double t)
//    {
//        double it = 1.0 - t;
//        switch (i)
//        {
//        case 0:
//            return it * it * it;
//        case 1:
//            return 3.0 * t * it * it;
//        case 2:
//            return 3.0 * t * t * it;
//        case 3:
//            return t * t * t;
//        default:
//            return 0.0;
//        }
//    }
//    __device__ inline double bernstein2(int i, double t)
//    {
//        double it = 1.0 - t;
//        switch (i)
//        {
//        case 0:
//            return it * it;
//        case 1:
//            return 2.0 * t * it;
//        case 2:
//            return t * t;
//        default:
//            return 0.0;
//        }
//    }
//    __device__ inline double3 getPoint(double u, double v)
//    {
//        double3 result = make_double3(0.0, 0.0, 0.0);
//        for (int i = 0; i < 4; i++)
//        {
//            double Bv = bernstein3(i, v);
//            for (int j = 0; j < 4; j++) 
//            {
//                result += Bv * bernstein3(j, u) * p[i][j];
//            }
//        }
//        return result;
//    }
//    __device__ inline double2 isPointOnBezier(const Ray& ray, HitRecord& record, double t_min, double t_max)
//    {
//        double len = 1.0;
//        double2 start = make_double2(0.0, 0.0);
//        int2 ij = make_int2(-1, -1);
//
//        for (int i = 0; i < 6; i++)
//        {
//            ij = iter(start, len, ray, record, t_min, t_max);
//            if (ij.x >= 0 && ij.y >= 0)
//            {
//                len *= 0.5;
//                start.x += len * ij.x;
//                start.y += len * ij.y;
//            }
//            else
//            {
//                return make_double2(-1.0, -1.0);
//            }
//        }
//        return make_double2(start.x, start.y);
//    }
//    __device__ inline int2 iter(double2 start, double len, const Ray& ray, HitRecord& record, double t_min, double t_max)
//    {
//        double delx = 0.5 * len;
//        for (int i = 0; i < 2; i++)
//        {
//            double u = start.x + i * delx;
//            for (int j = 0; j < 2; j++)
//            {
//                double v = start.y + j * delx;
//                double3 p0 = getPoint(u, v);
//                double3 p1 = getPoint(u + delx, v);
//                double3 p2 = getPoint(u + delx, v + delx);
//                double3 p3 = getPoint(u, v + delx);
//                if (hitTriangle(p0, p3, p1, ray, record, t_min, t_max) || hitTriangle(p1, p3, p2, ray, record, t_min, t_max))
//                    return make_int2(i, j);
//            }
//        }
//        return make_int2(-1, -1);
//    }
//    __device__ inline bool hitTriangle(const double3 p0, const double3 p1, const double3 p2, const Ray& ray, HitRecord& record, double t_min, double t_max)
//    {
//        const double EPSILON = 1e-8;
//
//        double3 edge1 = p1 - p0;
//        double3 edge2 = p2 - p0;
//
//        double3 h = Cross(ray.direction, edge2);
//        double a = Dot(edge1, h);
//        // parrallel
//        if (fabs(a) < EPSILON)
//            return false;
//
//        double f = 1.0 / a;
//        double3 s = ray.origin - p0;
//        double u = f * Dot(s, h);
//        if (u < 0.0 || u > 1.0)
//            return false;
//
//        double3 q = Cross(s, edge1);
//        double v = f * Dot(ray.direction, q);
//        if (v < 0.0 || u + v > 1.0)
//            return false;
//
//        double t = f * Dot(edge2, q);
//        if (t < t_min || t > t_max)
//            return false;
//
//        record.hitPos = ray.at(t);
//        record.t = t;
//        record.material = &material;
//        double3 outwardNoraml = Unit(Cross(edge1, edge2));
//        record.setFaceNormal(ray, outwardNoraml);
//
//        return true;
//    }
//};

class Light
{
public:
    double3 center;
    double width;
    double height;
    double3 normal;
    double3 edgeU;
    double3 edgeV;
    Material material;
    bool visible;

public:
    __host__ __device__ Light(double3 _center, double _width, double _height, double3 _normal, double3 _color, bool _visible = false)
        : center(_center), width(_width), height(_height), normal(_normal), material(_color, 1.0, 1.0, MaterialType::M_LIGHT), visible(_visible)
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
        // ! important change, origin version: t < 0 which is fault
        if (t < t_min || t > t_max)
            return false;

        double3 hitPoint = ray.at(t);
        double3 diff = hitPoint - center;
        double u = Dot(diff, edgeU);
        double v = Dot(diff, edgeV);
        if (fabs(u) <= width / 2.0 && fabs(v) <= height / 2.0)
        {
            record.hitPos = hitPoint;
            record.t = t;
            record.material = &material;
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
        //Bzeier bzeier;
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
    //// Bzeier constructor
    //__host__ __device__ Hittable(const Bzeier& b)
    //    : type(ObjectType::BEZIER), bzeier(b)
    //{
    //    center = (b.p[0][0] + b.p[0][3] + b.p[3][0] + b.p[3][3]) / 4.0;

    //    double3 minP = b.p[0][0];
    //    double3 maxP = b.p[0][0];
    //    for (int i = 0; i < 4; ++i) {
    //        for (int j = 0; j < 4; ++j) {
    //            minP.x = mMin(minP.x, b.p[i][j].x);
    //            minP.y = mMin(minP.y, b.p[i][j].y);
    //            minP.z = mMin(minP.z, b.p[i][j].z);

    //            maxP.x = mMax(maxP.x, b.p[i][j].x);
    //            maxP.y = mMax(maxP.y, b.p[i][j].y);
    //            maxP.z = mMax(maxP.z, b.p[i][j].z);
    //        }
    //    }
    //    aabb = AABB(minP, maxP).pad();
    //    printf("(%f %f %f)-(%f %f %f)\n", aabb.min.x, aabb.min.y, aabb.min.z, aabb.max.x, aabb.max.y, aabb.max.z);
    //}

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
        //case ObjectType::BEZIER:
        //    return bzeier.hit(ray, record, t_min, t_max);
        case ObjectType::NONE:
        default:
            return false;
        }
    }
};