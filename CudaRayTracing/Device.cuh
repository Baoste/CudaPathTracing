#pragma once

#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "BVH.cuh"
#include "ObjLoader.h"

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

__global__ inline void registerDevice(Hittable** d_objs, unsigned int* d_objPtr, int count)
{
    printf("Sign d_objs on device...\n");
    *d_objs = (Hittable*)malloc(count * sizeof(Hittable));  // raw allocation
    *d_objPtr = 0;  // reset the object pointer
}

__global__ inline void allocateSphereOnDevice(Hittable** d_objs, unsigned int* d_objPtr, double3 center, double r, double3 color)
{
    printf("Allocate Sphere (%f, %f, %f) on device...\n", center.x, center.y, center.z);
    new (&(*d_objs)[(*d_objPtr)++]) Hittable(Sphere(center, r, color));
}

__global__ inline void allocateLightsOnDevice(Hittable** d_objs, unsigned int* d_objPtr, unsigned int* d_lightsIndex, const int size)
{
    printf("Allocate %d Lights on device...\n", size);
    // TODO User-defined constructor for Hittable
    //for (int i = 0; i < size; i++)
    //{
    //     new (&(*d_objs)[*d_objPtr]) Hittable(Light( ...
    //     d_lightsIndex[i] = *d_objPtr;
    //     (*d_objPtr)++;
    //}
    new (&(*d_objs)[*d_objPtr]) Hittable(Light(make_double3(0, 4.0, 0), 2.0, 2.0, make_double3(0, -1.0, 0), make_double3(5.0, 1.0, 1.0)));
    d_lightsIndex[0] = (*d_objPtr)++;
    new (&(*d_objs)[*d_objPtr]) Hittable(Light(make_double3(0.0, 4.0, 4.0), 1.0, 1.0, make_double3(0.0, -1.0, -1.0), make_double3(1.0, 10.0, 10.0)));
    d_lightsIndex[1] = (*d_objPtr)++;
    new (&(*d_objs)[*d_objPtr]) Hittable(Light(make_double3(-2.0, 2.0, -9.9), 2.0, 2.0, make_double3(0.0, 0.0, 1.0), make_double3(20.0, 20.0, 20.0), true));
    d_lightsIndex[2] = (*d_objPtr)++;
}

__global__ inline void allocateMeshesOnDevice(Hittable** d_objs, unsigned int* d_objPtr, MeshTriangle* d_triangles, const int size)
{
    printf("Allocate %d Meshes on device...\n", size);
    // TODO need to delete
    new (&(*d_objs)[(*d_objPtr)++]) Hittable(Triangle(make_double3(-100.0, 0.0, -10.0), make_double3(-100.0, 0.0, 10.0), make_double3(100.0, 0.0, -10.0), make_double3(0.8, 0.8, 0.8)));
    new (&(*d_objs)[(*d_objPtr)++]) Hittable(Triangle(make_double3(100.0, 0.0, -10.0), make_double3(-100.0, 0.0, 10.0), make_double3(100.0, 0.0, 10.0), make_double3(0.8, 0.8, 0.8)));
    new (&(*d_objs)[(*d_objPtr)++]) Hittable(Triangle(make_double3(-100.0, 10.0, -10.0), make_double3(-100.0, 0.0, -10.0), make_double3(100.0, 0.0, -10.0), make_double3(0.8, 0.2, 0.1)));
    new (&(*d_objs)[(*d_objPtr)++]) Hittable(Triangle(make_double3(100.0, 0.0, -10.0), make_double3(100.0, 10.0, -10.0), make_double3(-100.0, 10.0, -10.0), make_double3(0.8, 0.2, 0.1)));
    
    for (int i = 0; i < size; i++)
    {
        new (&(*d_objs)[(*d_objPtr)++]) Hittable(Triangle(
            d_triangles[i].p0,
            d_triangles[i].p1,
            d_triangles[i].p2,
            make_double3(0.8, 0.8, 0.8),
            true
        ));
    }
}

__global__ inline void allocateClothToDevice(Hittable** d_objs, unsigned int* d_objPtr, unsigned int* d_clothPtr, double* d_x, int* d_idx, const int size)
{
    *d_clothPtr = *d_objPtr;  // record the first cloth object pointer
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            int quad_id = 6 * (i * size + j);
            new (&(*d_objs)[2 * (i * size + j) + 0 + *d_clothPtr]) Hittable(Triangle(
                make_double3(d_x[d_idx[quad_id + 0] * 3 + 0], d_x[d_idx[quad_id + 0] * 3 + 1], d_x[d_idx[quad_id + 0] * 3 + 2]),
                make_double3(d_x[d_idx[quad_id + 1] * 3 + 0], d_x[d_idx[quad_id + 1] * 3 + 1], d_x[d_idx[quad_id + 1] * 3 + 2]),
                make_double3(d_x[d_idx[quad_id + 2] * 3 + 0], d_x[d_idx[quad_id + 2] * 3 + 1], d_x[d_idx[quad_id + 2] * 3 + 2]),
                make_double3(0.5, 0.2, 0.1)
            ));
            (*d_objPtr)++;
            new (&(*d_objs)[2 * (i * size + j) + 1 + *d_clothPtr]) Hittable(Triangle(
                make_double3(d_x[d_idx[quad_id + 3] * 3 + 0], d_x[d_idx[quad_id + 3] * 3 + 1], d_x[d_idx[quad_id + 3] * 3 + 2]),
                make_double3(d_x[d_idx[quad_id + 4] * 3 + 0], d_x[d_idx[quad_id + 4] * 3 + 1], d_x[d_idx[quad_id + 4] * 3 + 2]),
                make_double3(d_x[d_idx[quad_id + 5] * 3 + 0], d_x[d_idx[quad_id + 5] * 3 + 1], d_x[d_idx[quad_id + 5] * 3 + 2]),
                make_double3(0.5, 0.2, 0.1)
            ));
            (*d_objPtr)++;
        }
    }
}

__global__ inline void updateClothToDevice(Hittable** d_objs, unsigned int* d_clothPtr, double* d_x, int* d_idx, const int size)
{
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            int quad_id = 6 * (i * size + j);
            new (&(*d_objs)[2 * (i * size + j) + 0 + *d_clothPtr]) Hittable(Triangle(
                make_double3(d_x[d_idx[quad_id + 0] * 3 + 0], d_x[d_idx[quad_id + 0] * 3 + 1], d_x[d_idx[quad_id + 0] * 3 + 2]),
                make_double3(d_x[d_idx[quad_id + 1] * 3 + 0], d_x[d_idx[quad_id + 1] * 3 + 1], d_x[d_idx[quad_id + 1] * 3 + 2]),
                make_double3(d_x[d_idx[quad_id + 2] * 3 + 0], d_x[d_idx[quad_id + 2] * 3 + 1], d_x[d_idx[quad_id + 2] * 3 + 2]),
                make_double3(0.5, 0.2, 0.1)
            ));
            new (&(*d_objs)[2 * (i * size + j) + 1 + *d_clothPtr]) Hittable(Triangle(
                make_double3(d_x[d_idx[quad_id + 3] * 3 + 0], d_x[d_idx[quad_id + 3] * 3 + 1], d_x[d_idx[quad_id + 3] * 3 + 2]),
                make_double3(d_x[d_idx[quad_id + 4] * 3 + 0], d_x[d_idx[quad_id + 4] * 3 + 1], d_x[d_idx[quad_id + 4] * 3 + 2]),
                make_double3(d_x[d_idx[quad_id + 5] * 3 + 0], d_x[d_idx[quad_id + 5] * 3 + 1], d_x[d_idx[quad_id + 5] * 3 + 2]),
                make_double3(0.5, 0.2, 0.1)
            ));
        }
    }
}

__global__ inline void generateMortonCodes(Hittable** d_objs, unsigned int* d_mortons, unsigned int* d_objsIdx, const int size)
{
    for (int i = 0; i < size; i++)
    {
        // printf("Object %d: Center: (%f, %f, %f)\n", i, (*d_b)[i].center.x, (*d_b)[i].center.y, (*d_b)[i].center.z);
        d_mortons[i] = morton3D((*d_objs)[i].center);
        d_objsIdx[i] = i;
    }
}

__global__ inline void generateBVH(Hittable** d_objs, Node* leafNodes, Node* internalNodes, unsigned int* d_mortons, unsigned int* d_objsIdx, const int size)
{
    generateHierarchy(d_objs, leafNodes, internalNodes, d_mortons, d_objsIdx, size);
}

class Device
{
public:
    Hittable** d_objs;
    unsigned int* d_objPtr;          // refer the last object in the d_objs array
    unsigned int* d_clothPtr;          // refer the first cloth in the d_objs array

public:
    Device()
    {
        cudaMalloc((void**)&d_objs, sizeof(Hittable**));
        cudaMalloc((void**)&d_objPtr, sizeof(unsigned int));
        cudaMalloc((void**)&d_clothPtr, sizeof(unsigned int));
        registerDevice << <1, 1 >> > (d_objs, d_objPtr, 20000);
    }
    void buildBVH(Hittable** d_objs, Node* leafNodes, Node* internalNodes, unsigned int* d_mortons, unsigned int* d_objsIdx, const int size)
    {
        generateMortonCodes << <1, 1 >> > (d_objs, d_mortons, d_objsIdx, size);
        checkCudaErrors(cudaDeviceSynchronize());
        // sort
        thrust::device_ptr<unsigned int> d_keys(d_mortons);
        thrust::device_ptr<unsigned int> d_values(d_objsIdx);
        thrust::sort_by_key(d_keys, d_keys + size, d_values);

        generateBVH << <1, 1 >> > (d_objs, leafNodes, internalNodes, d_mortons, d_objsIdx, size);
        checkCudaErrors(cudaDeviceSynchronize());
    }
};