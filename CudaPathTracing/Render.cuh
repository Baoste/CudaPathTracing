#pragma once

#include "Ray.cuh"
#include "Camera.cuh"
#include "Hittable.cuh"
#include "BVH.cuh"

#define KERNEL_RADIUS 16
#define KERNEL_SIZE (2 * KERNEL_RADIUS + 1)

enum RenderType { REAL_TIME_NOT_SAMPLE, REAL_TIME, STATIC, NORMAL, DEPTH };

__global__ void clear(double3* pic, int max_x, int max_y);

__global__ void render(uchar4* devPtr, double3* pic, double3* picPrevious, double3* picBeforeGussian, double4* gBuffer, double3* gBufferPosition,
    const Camera* camera, unsigned int* lightsIndex, Hittable* objs, Node* internalNodes, int lightsCount,
    int max_x, int max_y, int sampleCount, double t, RenderType rederType);

__global__ void copyPic(double3* target, double3* source, int max_x, int max_y);

__global__ void gaussianSeparate(double3* pic, double3* picBeforeGussian, double4* gBuffer, double sigmaG, double sigmaR, double sigmaN, double sigmaD, int max_x, int max_y, bool isHorizontal);

__global__ void gaussian(double3* pic, double3* picBeforeGussian, double4* gBuffer, double sigmaG, double sigmaR, double sigmaN, double sigmaD, int max_x, int max_y);

__global__ void addPrevious(double3* pic, double3* picPrevious, double3* gBufferPosition, const Camera* camera, int max_x, int max_y);

__global__ void pic2RGBW(uchar4* devPtr, double3* pic, int max_x, int max_y);

__global__ void getObject(Hittable* objs, const Camera* camera, Node* internalNodes, int* selectPtr, const int x, const int y);

__global__ void changeMaterial(Hittable* objs, const int start, const int end, const double roughness, const double metallic, const bool glass);
