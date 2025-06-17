#pragma once

#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "BVH.cuh"
#include "ObjLoader.h"
#include "Cloth.cuh"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

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

__global__ inline void allocateOnDevice(Hittable** d_b, MeshTriangle* d_triangles, unsigned int* d_lightsIndex, double* d_x, int* d_idx, unsigned int* d_mortons, unsigned int* d_objsIdx, const size_t size, const size_t tri_size)
{
    printf("Allocating %d objects on device...\n", size);

    *d_b = (Hittable*)malloc(size * sizeof(Hittable));  // raw allocation

    // lights
    new (&(*d_b)[0]) Hittable(Light(make_double3(0, 4.0, 0), 2.0, 2.0, make_double3(0, -1.0, 0), make_double3(5.0, 1.0, 1.0)));
    new (&(*d_b)[1]) Hittable(Light(make_double3(0.0, 4.0, 4.0), 1.0, 1.0, make_double3(0.0, -1.0, -1.0), make_double3(1.0, 10.0, 10.0)));
    // index of the light source
    d_lightsIndex[0] = 0;
    d_lightsIndex[1] = 1;

    // floor
    new (&(*d_b)[2]) Hittable(Triangle(make_double3(-100.0, 0.0, -10.0), make_double3(-100.0, 0.0, 10.0), make_double3(100.0, 0.0, -10.0), make_double3(0.8, 0.8, 0.8)));
    new (&(*d_b)[3]) Hittable(Triangle(make_double3(100.0, 0.0, -10.0), make_double3(-100.0, 0.0, 10.0), make_double3(100.0, 0.0, 10.0), make_double3(0.8, 0.8, 0.8)));
    // obj
    for (size_t i = 0; i < size - 4 - tri_size; i++)
    {
        new (&(*d_b)[i + 4]) Hittable(Triangle(
            d_triangles[i].p0,
            d_triangles[i].p1,
            d_triangles[i].p2,
            make_double3(0.8, 0.8, 0.8)
        ));
    }
    for (int i = 0; i < division - 1; i++)
    {
        for (int j = 0; j < division - 1; j++)
        {
            int quad_id = 6 * (i * (division - 1) + j);
            new (&(*d_b)[2 * (i * (division - 1) + j) + 0 + size - tri_size]) Hittable(Triangle(
                make_double3(d_x[d_idx[quad_id + 0] * 3 + 0], d_x[d_idx[quad_id + 0] * 3 + 1], d_x[d_idx[quad_id + 0] * 3 + 2]),
                make_double3(d_x[d_idx[quad_id + 1] * 3 + 0], d_x[d_idx[quad_id + 1] * 3 + 1], d_x[d_idx[quad_id + 1] * 3 + 2]),
                make_double3(d_x[d_idx[quad_id + 2] * 3 + 0], d_x[d_idx[quad_id + 2] * 3 + 1], d_x[d_idx[quad_id + 2] * 3 + 2]),
                make_double3(0.8, 0.8, 0.8)
            ));
            new (&(*d_b)[2 * (i * (division - 1) + j) + 1 + size - tri_size]) Hittable(Triangle(
                make_double3(d_x[d_idx[quad_id + 3] * 3 + 0], d_x[d_idx[quad_id + 3] * 3 + 1], d_x[d_idx[quad_id + 3] * 3 + 2]),
                make_double3(d_x[d_idx[quad_id + 4] * 3 + 0], d_x[d_idx[quad_id + 4] * 3 + 1], d_x[d_idx[quad_id + 4] * 3 + 2]),
                make_double3(d_x[d_idx[quad_id + 5] * 3 + 0], d_x[d_idx[quad_id + 5] * 3 + 1], d_x[d_idx[quad_id + 5] * 3 + 2]),
                make_double3(0.8, 0.8, 0.8)
            ));
        }
    }

    for (size_t i = 0; i < size; i++)
    {
        // printf("Object %d: Center: (%f, %f, %f)\n", i, (*d_b)[i].center.x, (*d_b)[i].center.y, (*d_b)[i].center.z);
        d_mortons[i] = morton3D((*d_b)[i].center);
        d_objsIdx[i] = i;
    }
}

__global__ inline void buildBVH(Hittable** d_objs, Node* leafNodes, Node* internalNodes, unsigned int* sortedMortonCodes, unsigned int* sortedObjectIDs, int objsCount)
{
    generateHierarchy(d_objs, leafNodes, internalNodes, sortedMortonCodes, sortedObjectIDs, objsCount);
}

class Scene
{
public:
    Hittable** d_objs;
    unsigned int* d_lightsIndex;
    Node* internalNodes;
    size_t objsCount;
    size_t lightsCount;

private:
    Node* leafNodes;
    unsigned int* d_mortons;
    unsigned int* d_objIdx;
    size_t allCount;

public:
    Scene()
    {
        objsCount = 0;
        lightsCount = 0;
        allCount = 0;
    }
    ~Scene()
    {
        cudaFree(d_mortons);
        cudaFree(d_objIdx);
        cudaFree(internalNodes);
        cudaFree(leafNodes);
    }
    void init()
    {
        Mesh mesh;
        mesh.loadFromFile("C:/Users/59409/Downloads/teapot.obj");
        int num = mesh.triangles.size();
        MeshTriangle* d_triangles;
        cudaMalloc((void**)&d_triangles, num * sizeof(MeshTriangle));
        cudaMemcpy(d_triangles, mesh.triangles.data(), num * sizeof(MeshTriangle), cudaMemcpyHostToDevice);

        Cloth cloth;
        cloth.initialize();
        for (size_t i = 0; i < 1200; i++)
        {
            cloth.Update();
        }
        double* d_x;
        cudaMalloc((void**)&d_x, numParticles * 3 * sizeof(double));
        cudaMemcpy(d_x, cloth.X, numParticles * 3 * sizeof(double), cudaMemcpyHostToDevice);
        int* d_idx;
        cudaMalloc((void**)&d_idx, 6 * (division - 1) * (division - 1) * sizeof(int));
        cudaMemcpy(d_idx, cloth.triangleIdx, 6 * (division - 1) * (division - 1) * sizeof(int), cudaMemcpyHostToDevice);

        objsCount = 2 + num + 2 * (division - 1) * (division - 1);
        lightsCount = 2;
        allCount = objsCount + lightsCount;
        cudaMalloc((void**)&d_objs, sizeof(Hittable**));
        cudaMalloc((void**)&d_lightsIndex, lightsCount * sizeof(unsigned int));
        cudaMalloc((void**)&d_mortons, allCount * sizeof(unsigned int));
        cudaMalloc((void**)&d_objIdx, allCount * sizeof(unsigned int));
        allocateOnDevice << < 1, 1 >> > (d_objs, d_triangles, d_lightsIndex, d_x, d_idx, d_mortons, d_objIdx, allCount, 2 * (division - 1) * (division - 1));
        checkCudaErrors(cudaDeviceSynchronize());

        // 包装为 device_ptr
        thrust::device_ptr<unsigned int> d_keys(d_mortons);
        thrust::device_ptr<unsigned int> d_values(d_objIdx);

        // 排序（按 key 排序，同时 value 做同样重排）
        thrust::sort_by_key(d_keys, d_keys + allCount, d_values);

        cudaMalloc((void**)&internalNodes, (allCount - 1) * sizeof(Node));
        cudaMalloc((void**)&leafNodes, allCount * sizeof(Node));
        buildBVH << < 1, 1 >> > (d_objs, leafNodes, internalNodes, d_mortons, d_objIdx, allCount);
        checkCudaErrors(cudaDeviceSynchronize());
    }
};