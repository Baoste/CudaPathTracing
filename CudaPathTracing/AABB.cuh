#pragma once

class AABB
{
public:
    double3 min;
    double3 max;

public:
    __host__ __device__ AABB() : min(make_double3(INF, INF, INF)), max(make_double3(-INF, -INF, -INF)) {}
    __host__ __device__ AABB(const double3 p0, const double3 p1)
    {
        min.x = mMin(p0.x, p1.x);
        min.y = mMin(p0.y, p1.y);
        min.z = mMin(p0.z, p1.z);
        max.x = mMax(p0.x, p1.x);
        max.y = mMax(p0.y, p1.y);
        max.z = mMax(p0.z, p1.z);
    }
    __host__ __device__ AABB(const AABB& box0, const AABB& box1)
    {
        min.x = mMin(box0.min.x, box1.min.x);
        min.y = mMin(box0.min.y, box1.min.y);
        min.z = mMin(box0.min.z, box1.min.z);
        max.x = mMax(box0.max.x, box1.max.x);
        max.y = mMax(box0.max.y, box1.max.y);
        max.z = mMax(box0.max.z, box1.max.z);
    }

    __host__ __device__ inline AABB pad()
    {
        double delta = 0.0001;
        min.x -= delta;
        min.y -= delta;
        min.z -= delta;
        max.x += delta;
        max.y += delta;
        max.z += delta;
        return AABB(min, max);
    }
};