#pragma once

#include "Ray.cuh"
#include "Camera.cuh"
#include "Hittable.cuh"
#include "BVH.cuh"

__global__ void render(uchar4* ptr, const Camera* camera, unsigned int* lightsIndex, Hittable* objs, Node* internalNodes, int lightsCount,
    int max_x, int max_y, int sampleCount, double t);

__global__ void getObject(Hittable* objs, const Camera* camera, Node* internalNodes, int* selectPtr, const int x, const int y);

__global__ void changeMaterial(Hittable* objs, const int start, const int end, const double roughness, const double metallic);