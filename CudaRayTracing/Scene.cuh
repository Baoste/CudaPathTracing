#pragma once

#include "Device.cuh"
#include "Cloth.cuh"
#include "PhysicsAcc.cuh"


class Scene
{
public:
    Device device;
    unsigned int* d_lightsIndex;
    Node* internalNodes;
    Node* leafNodes;

    Node* internalNodesExceptCloth;
    Node* leafNodesExceptCloth;
    
    int allCount;
    int lightsCount;

private:
    unsigned int* d_mortons;
    unsigned int* d_objIdx;

    unsigned int* d_mortonsExceptCloth;
    unsigned int* d_objIdxExceptCloth;
    double* d_X, * d_V, * d_F, * d_M, * d_L;
    int* d_edgeIdx, * d_triIdx;

    Cloth cloth;

public:
    Scene() : cloth(3.0, make_double3(2.0, 2.3, 2.0))
    {
        allCount = 0;
        lightsCount = 3;
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
        cudaMalloc((void**)&d_lightsIndex, lightsCount * sizeof(unsigned int));
        allocateLightsOnDevice << < 1, 1 >> > (device.d_objs, device.d_objPtr, d_lightsIndex, lightsCount);
        checkCudaErrors(cudaDeviceSynchronize());

        allocateSphereOnDevice << < 1, 1 >> > (device.d_objs, device.d_objPtr, make_double3(2.0, 1.0, 2.0), 1.0, make_double3(0.2, 0.8, 0.1));
        checkCudaErrors(cudaDeviceSynchronize());

        Mesh mesh;
        mesh.loadFromFile("C:/Users/59409/Downloads/teapot.obj");
        mesh.transform(make_double3(-1.0, 0.0, -1.0));
        int num = mesh.triangles.size();
        MeshTriangle* d_triangles;
        cudaMalloc((void**)&d_triangles, num * sizeof(MeshTriangle));
        cudaMemcpy(d_triangles, mesh.triangles.data(), num * sizeof(MeshTriangle), cudaMemcpyHostToDevice);
        allocateMeshesOnDevice << < 1, 1 >> > (device.d_objs, device.d_objPtr, d_triangles, num);
        checkCudaErrors(cudaDeviceSynchronize());

        allCount = lightsCount + num + 4;     // lights + mesh triangles + floors * 2
        cudaMalloc((void**)&d_mortonsExceptCloth, allCount * sizeof(unsigned int));
        cudaMalloc((void**)&d_objIdxExceptCloth, allCount * sizeof(unsigned int));
        cudaMalloc((void**)&leafNodesExceptCloth, allCount * sizeof(Node));
        cudaMalloc((void**)&internalNodesExceptCloth, (allCount - 1) * sizeof(Node));
        device.buildBVH(device.d_objs, leafNodesExceptCloth, internalNodesExceptCloth, d_mortonsExceptCloth, d_objIdxExceptCloth, allCount);
        checkCudaErrors(cudaDeviceSynchronize());

        // allocate cloth triangles to device
        cloth.initialize();        
        cudaMalloc((void**)&d_X, numPartAxis * sizeof(double));
        cudaMemcpy(d_X, cloth.X, numPartAxis * sizeof(double), cudaMemcpyHostToDevice);
        cudaMalloc((void**)&d_V, numPartAxis * sizeof(double));
        cudaMemcpy(d_V, cloth.V, numPartAxis * sizeof(double), cudaMemcpyHostToDevice);
        cudaMalloc((void**)&d_F, numPartAxis * sizeof(double));
        cudaMemcpy(d_F, cloth.F, numPartAxis * sizeof(double), cudaMemcpyHostToDevice);
        cudaMalloc((void**)&d_M, numPartAxis * numPartAxis * sizeof(double));
        cudaMemcpy(d_M, cloth.M, numPartAxis * numPartAxis * sizeof(double), cudaMemcpyHostToDevice);
        cudaMalloc((void**)&d_L, numParticles * numParticles * sizeof(double));
        cudaMemcpy(d_L, cloth.L, numParticles * numParticles * sizeof(double), cudaMemcpyHostToDevice);
        cudaMalloc((void**)&d_edgeIdx, (6 * division * division - 12 * division + 6) * 2 * sizeof(int));
        cudaMemcpy(d_edgeIdx, cloth.edgeIdx, (6 * division * division - 12 * division + 6) * 2 * sizeof(int), cudaMemcpyHostToDevice);
        cudaMalloc((void**)&d_triIdx, 6 * (division - 1) * (division - 1) * sizeof(int));
        cudaMemcpy(d_triIdx, cloth.triangleIdx, 6 * (division - 1) * (division - 1) * sizeof(int), cudaMemcpyHostToDevice);
        allocateClothToDevice << < 1, 1 >> > (device.d_objs, device.d_objPtr, device.d_clothPtr, d_X, d_triIdx, division - 1);
        checkCudaErrors(cudaDeviceSynchronize());
        
        size_t clothTriCount = 2 * (division - 1) * (division - 1);
        allCount += clothTriCount;

        cudaMalloc((void**)&d_mortons, allCount * sizeof(unsigned int));
        cudaMalloc((void**)&d_objIdx, allCount * sizeof(unsigned int));
        cudaMalloc((void**)&internalNodes, (allCount - 1) * sizeof(Node));
        cudaMalloc((void**)&leafNodes, allCount * sizeof(Node));
        
        double simTime = 0.1;
        printf("Start sim physics for %f seconds...\n", simTime);
        while (cloth.simTime < simTime)
        {
            Update();
            break;
        }
    }

    void Update()
    {
        cloth.simTime += cloth.dt;

        int threadsPerBlock = 512;
        int blocks = (numParticles + threadsPerBlock - 1) / threadsPerBlock;
        PhysicsUpdate << <blocks, threadsPerBlock >> > (cloth.dt, d_X, d_V, d_F, d_M, d_L, d_edgeIdx, internalNodesExceptCloth, device.d_objs, numParticles);
        checkCudaErrors(cudaDeviceSynchronize());

        // allocate cloth triangles to device
        //cudaFree(d_V);
        //cudaFree(d_F);
        //cudaFree(d_M);
        //cudaFree(d_L);
        //cudaFree(d_edgeIdx);
        //cudaFree(d_mortonsExceptCloth);
        //cudaFree(d_objIdxExceptCloth);
        //cudaFree(leafNodesExceptCloth);
        //cudaFree(internalNodesExceptCloth);

        updateClothToDevice << < 1, 1 >> > (device.d_objs, device.d_clothPtr, d_X, d_triIdx, division - 1);
        checkCudaErrors(cudaDeviceSynchronize());

        device.buildBVH(device.d_objs, leafNodes, internalNodes, d_mortons, d_objIdx, allCount);
        checkCudaErrors(cudaDeviceSynchronize());

    }
};