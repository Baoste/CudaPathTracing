#pragma once

#include "Ray.cuh"
#include "Camera.cuh"
#include "Hittable.cuh"
#include "BVH.cuh"

__global__ void render(unsigned char* cb, const Camera* camera, unsigned int* lightsIndex, Hittable** objs, Node* internalNodes, int lightsCount, int max_x, int max_y, int sampleCount, double t);