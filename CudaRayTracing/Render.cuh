#pragma once

#include "Ray.cuh"
#include "Camera.cuh"
#include "Hittable.cuh"

__device__ bool isHitAnything(Hittable** objs, int obj_count, const Ray& ray, HitRecord& record);
__global__ void render(unsigned char* cb, Camera* camera, Hittable** objs, int obj_count, int max_x, int max_y, double t);