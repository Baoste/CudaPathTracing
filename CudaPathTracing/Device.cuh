#pragma once

#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "BVH.cuh"
#include "ObjLoader.cuh"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__)
inline void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line)
{
    if (result)
    {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}

const int MAXTRIANGLE = 1000000;

__global__ inline void registerDevice(unsigned int* d_objPtr, unsigned int* d_lightPtr)
{
    *d_objPtr = 0;  // reset the object pointer
    *d_lightPtr = 0;  // reset the object pointer
}

__global__ inline void allocateLightOnDevice(Hittable* d_objs, unsigned int* d_objPtr, unsigned int* d_lightsIndex, unsigned int* d_lightPtr, double3 position, double width, double height, double3 normal, double3 color, bool visible = false)
{
    new (&d_objs[*d_objPtr]) Hittable(Light(position, width, height, normal, color, visible));
    d_lightsIndex[(*d_lightPtr)++] = (*d_objPtr)++;
}

__global__ inline void allocateSphereOnDevice(Hittable* d_objs, unsigned int* d_objPtr, double3 center, double r, double3 color, double alphaX, double alphaY, MaterialType type)
{
    new (&d_objs[(*d_objPtr)++]) Hittable(Sphere(center, r, color, alphaX, alphaY, type));
}

__global__ inline void allocateFloorOnDevice(Hittable* d_objs, unsigned int* d_objPtr, double3 lt , double3 rt, double3 lb, double3 rb, double3 color, double alphaX, double alphaY, MaterialType type)
{
    new (&d_objs[(*d_objPtr)++]) Hittable(Triangle(lt, lb, rt, color, alphaX, alphaY, type));
    new (&d_objs[(*d_objPtr)++]) Hittable(Triangle(rt, lb, rb, color, alphaX, alphaY, type));
}

//__global__ inline void allocateBzeierOnDevice(Hittable* d_objs, unsigned int* d_objPtr, 
//    const double3 p0, const double3 p1, const double3 p2, const double3 p3,
//    const double3 p4, const double3 p5, const double3 p6, const double3 p7,
//    const double3 p8, const double3 p9, const double3 p10, const double3 p11,
//    const double3 p12, const double3 p13, const double3 p14, const double3 p15,
//    const double3 color, double alphaX, double alphaY)
//{
//    new (&d_objs[(*d_objPtr)++]) Hittable(Bzeier(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, color, alphaX, alphaY));
//}

__global__ inline void allocateMeshesOnDevice(Hittable* d_objs, unsigned int* d_objPtr, MeshTriangle* d_triangles, unsigned char* d_image, int width, int height, int channels, MeshUV* d_uvs, double3 color, double alphaX, double alphaY, MaterialType type, const int size)
{
    for (int i = 0; i < size; i++)
    {
        new (&d_objs[(*d_objPtr)++]) Hittable(Triangle(
            d_triangles[i].p0,
            d_triangles[i].p1,
            d_triangles[i].p2,
            color,
            alphaX, alphaY,
            type,
            d_uvs[i].p0,
            d_uvs[i].p1,
            d_uvs[i].p2,
            d_image,
            width,
            height,
            channels
        ));
    }
}

__global__ inline void allocateClothToDevice(Hittable* d_objs, unsigned int* d_objPtr, unsigned int* d_clothPtr, double* d_x, int* d_idx, const int size)
{
    *d_clothPtr = *d_objPtr;  // record the first cloth object pointer
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            int quad_id = 6 * (i * size + j);
            new (&d_objs[2 * (i * size + j) + 0 + *d_clothPtr]) Hittable(Triangle(
                make_double3(d_x[d_idx[quad_id + 0] * 3 + 0], d_x[d_idx[quad_id + 0] * 3 + 1], d_x[d_idx[quad_id + 0] * 3 + 2]),
                make_double3(d_x[d_idx[quad_id + 1] * 3 + 0], d_x[d_idx[quad_id + 1] * 3 + 1], d_x[d_idx[quad_id + 1] * 3 + 2]),
                make_double3(d_x[d_idx[quad_id + 2] * 3 + 0], d_x[d_idx[quad_id + 2] * 3 + 1], d_x[d_idx[quad_id + 2] * 3 + 2]),
                make_double3(0.5, 0.2, 0.1),
                0.5, 0.5
            ));
            (*d_objPtr)++;
            new (&d_objs[2 * (i * size + j) + 1 + *d_clothPtr]) Hittable(Triangle(
                make_double3(d_x[d_idx[quad_id + 3] * 3 + 0], d_x[d_idx[quad_id + 3] * 3 + 1], d_x[d_idx[quad_id + 3] * 3 + 2]),
                make_double3(d_x[d_idx[quad_id + 4] * 3 + 0], d_x[d_idx[quad_id + 4] * 3 + 1], d_x[d_idx[quad_id + 4] * 3 + 2]),
                make_double3(d_x[d_idx[quad_id + 5] * 3 + 0], d_x[d_idx[quad_id + 5] * 3 + 1], d_x[d_idx[quad_id + 5] * 3 + 2]),
                make_double3(0.5, 0.2, 0.1),
                0.5, 0.5
            ));
            (*d_objPtr)++;
        }
    }
}

__global__ inline void updateClothToDevice(Hittable* d_objs, unsigned int* d_clothPtr, double* d_x, int* d_idx, const int size)
{
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            int quad_id = 6 * (i * size + j);
            new (&d_objs[2 * (i * size + j) + 0 + *d_clothPtr]) Hittable(Triangle(
                make_double3(d_x[d_idx[quad_id + 0] * 3 + 0], d_x[d_idx[quad_id + 0] * 3 + 1], d_x[d_idx[quad_id + 0] * 3 + 2]),
                make_double3(d_x[d_idx[quad_id + 1] * 3 + 0], d_x[d_idx[quad_id + 1] * 3 + 1], d_x[d_idx[quad_id + 1] * 3 + 2]),
                make_double3(d_x[d_idx[quad_id + 2] * 3 + 0], d_x[d_idx[quad_id + 2] * 3 + 1], d_x[d_idx[quad_id + 2] * 3 + 2]),
                make_double3(0.5, 0.2, 0.1),
                0.5, 0.5
            ));
            new (&d_objs[2 * (i * size + j) + 1 + *d_clothPtr]) Hittable(Triangle(
                make_double3(d_x[d_idx[quad_id + 3] * 3 + 0], d_x[d_idx[quad_id + 3] * 3 + 1], d_x[d_idx[quad_id + 3] * 3 + 2]),
                make_double3(d_x[d_idx[quad_id + 4] * 3 + 0], d_x[d_idx[quad_id + 4] * 3 + 1], d_x[d_idx[quad_id + 4] * 3 + 2]),
                make_double3(d_x[d_idx[quad_id + 5] * 3 + 0], d_x[d_idx[quad_id + 5] * 3 + 1], d_x[d_idx[quad_id + 5] * 3 + 2]),
                make_double3(0.5, 0.2, 0.1),
                0.5, 0.5
            ));
        }
    }
}

__global__ inline void generateMortonCodes(Hittable* d_objs, unsigned int* d_mortons, unsigned int* d_objsIdx, const int size, const double3 sceneMin, const double3 sceneMax)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        d_mortons[i] = morton3D(d_objs[i].center, sceneMin, sceneMax, i);
        d_objsIdx[i] = i;
    }
}

__global__ inline void generateBVH(Hittable* d_objs, Node* leafNodes, Node* internalNodes, unsigned int* d_mortons, unsigned int* d_objsIdx, const int size)
{
    generateHierarchy(d_objs, leafNodes, internalNodes, d_mortons, d_objsIdx, size);
}

class Device
{
public:
    Hittable* d_objs;
    unsigned int* d_objPtr;          // refer the last object in the d_objs array
    unsigned int* d_lightPtr;          // refer the last object in the d_objs array
    unsigned int* d_clothPtr;          // refer the first cloth in the d_objs array

public:
    Device()
    {
        cudaMalloc((void**)&d_objs, MAXTRIANGLE * sizeof(Hittable));
        cudaMalloc((void**)&d_objPtr, sizeof(unsigned int));
        cudaMalloc((void**)&d_lightPtr, sizeof(unsigned int));
        cudaMalloc((void**)&d_clothPtr, sizeof(unsigned int));
        registerDevice << <1, 1 >> > (d_objPtr, d_lightPtr);
    }
    void buildBVH(Hittable* d_objs, Node* leafNodes, Node* internalNodes, unsigned int* d_mortons, unsigned int* d_objsIdx, const int size, double3 minBoundary, double3 maxBoundary)
    {
        int threadsPerBlock = 256;
        int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;

        double3 expand = make_double3(1.0, 1.0, 1.0) * 7.0;
        double3 sceneMin = minBoundary - expand;
        double3 sceneMax = maxBoundary + expand;
        if (sceneMin.x > sceneMax.x)
        {
            sceneMin = make_double3(-20.0, -20.0, -20.0);
            sceneMax = make_double3(20.0, 20.0, 20.0);
        }
        printf("Scene Boundary: (%f, %f, %f) - (%f, %f, %f)\n", sceneMin.x, sceneMin.y, sceneMin.z, sceneMax.x, sceneMax.y, sceneMax.z);

        generateMortonCodes << <blocks, threadsPerBlock >> > (d_objs, d_mortons, d_objsIdx, size, sceneMin, sceneMax);
        checkCudaErrors(cudaDeviceSynchronize());
        // sort
        thrust::device_ptr<unsigned int> d_keys(d_mortons);
        thrust::device_ptr<unsigned int> d_values(d_objsIdx);
        thrust::sort_by_key(d_keys, d_keys + size, d_values);

        generateBVH << <1, 1 >> > (d_objs, leafNodes, internalNodes, d_mortons, d_objsIdx, size);
        checkCudaErrors(cudaDeviceSynchronize());
    }
};