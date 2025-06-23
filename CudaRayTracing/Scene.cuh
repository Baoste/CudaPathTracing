#pragma once

#include <vector>
#include <string>

#include "Device.cuh"
#include "Cloth.cuh"
#include "PhysicsAcc.cuh"

struct Object
{
    std::string name;
    unsigned int beginPtr;
    unsigned int endPtr;
};

class Scene
{
public:
    Device device;
    Camera* d_camera;
    std::vector<Object> objects;
    int* d_selectPtr;

public:
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
        lightsCount = 0;
        cudaMalloc((void**)&d_camera, sizeof(Camera));
        cudaMalloc((void**)&d_selectPtr, sizeof(int));
    }
    ~Scene()
    {
        cudaFree(d_mortons);
        cudaFree(d_objIdx);
        cudaFree(internalNodes);
        cudaFree(leafNodes);
        cudaFree(d_camera);
    }

    void init()
    {
        // light
        cudaMalloc((void**)&d_lightsIndex, 10 * sizeof(unsigned int));
        addOneLight(make_double3(0.0, 9.99, 0.0), 3.0, 3.0, make_double3(0.0, -1.0, 0.0), make_double3(20.0, 20.0, 20.0), true);
        addOneLight(make_double3(0.0, 0.0, 9.99), 3.0, 3.0, make_double3(0.0, 0.0, -1.0), make_double3(20.0, 20.0, 20.0), true);
        // addOneLight(make_double3(-2.0, 2.0, -9.9), 2.0, 2.0, make_double3(0.0, 0.0, 1.0), make_double3(20.0, 20.0, 20.0), true);

        // sphere
        addOneSphere(make_double3(2.0, 1.0, 2.0), 1.0, make_double3(0.5, 0.4, 0.6));
        //addOneSphere(make_double3(-2.0, 3.0, -2.0), 1.0, make_double3(1.0, 0.84, 0.0));

        // floor
        addFloor(make_double3(-6.0, 0.0, -6.0), make_double3(6.0, 0.0, -6.0), make_double3(-6.0, 0.0, 6.0), make_double3(6.0, 0.0, 6.0), make_double3(0.8, 0.8, 0.8));
        addFloor(make_double3(-6.0, 10.0, -6.0), make_double3(6.0, 10.0, -6.0), make_double3(-6.0, 0.0, -6.0), make_double3(6.0, 0.0, -6.0), make_double3(0.8, 0.8, 0.8));
        addFloor(make_double3(-6.0, 10.0, 6.0), make_double3(-6.0, 10.0, -6.0), make_double3(-6.0, 0.0, 6.0), make_double3(-6.0, 0.0, -6.0), make_double3(0.2, 0.8, 0.1));
        addFloor(make_double3(6.0, 10.0, -6.0), make_double3(6.0, 10.0, 6.0), make_double3(6.0, 0.0, -6.0), make_double3(6.0, 0.0, 6.0), make_double3(0.1, 0.2, 0.8));
        addFloor(make_double3(-6.0, 10.0, 6.0), make_double3(6.0, 10.0, 6.0), make_double3(-6.0, 10.0, -6.0), make_double3(6.0, 10.0, -6.0), make_double3(0.8, 0.8, 0.8));

        // mesh
        addMeshes("C:/Users/59409/Downloads/teapot.obj", make_double3(-2.0, 0.0, 2.0), make_double3(0.8, 0.8, 0.8), true, 0.7);
        addMeshes("C:/Users/59409/Downloads/bun_zipper.ply.obj", make_double3(-1.0, -0.6, -1.0), make_double3(0.8, 0.8, 0.8), false, 20.0);

        cudaMalloc((void**)&d_mortonsExceptCloth, allCount * sizeof(unsigned int));
        cudaMalloc((void**)&d_objIdxExceptCloth, allCount * sizeof(unsigned int));
        cudaMalloc((void**)&leafNodesExceptCloth, allCount * sizeof(Node));
        cudaMalloc((void**)&internalNodesExceptCloth, (allCount - 1) * sizeof(Node));
        device.buildBVH(device.d_objs, leafNodesExceptCloth, internalNodesExceptCloth, d_mortonsExceptCloth, d_objIdxExceptCloth, allCount);
        checkCudaErrors(cudaDeviceSynchronize());

        // allocate cloth triangles to device
        addCloth();

        cudaMalloc((void**)&d_mortons, allCount * sizeof(unsigned int));
        cudaMalloc((void**)&d_objIdx, allCount * sizeof(unsigned int));
        cudaMalloc((void**)&internalNodes, (allCount - 1) * sizeof(Node));
        cudaMalloc((void**)&leafNodes, allCount * sizeof(Node));

        double simTime = 1.6;
        printf("Start sim physics for %f seconds...\n", simTime);
        while (cloth.simTime < simTime)
        {
            Update();
            // break;
        }
    }

    void Update()
    {
        cloth.simTime += cloth.dt;

        int threadsPerBlock = 256;
        int blocks = (numParticles + threadsPerBlock - 1) / threadsPerBlock;
        PhysicsUpdate<<<blocks, threadsPerBlock>>>(cloth.dt, d_X, d_V, d_F, d_M, d_L, d_edgeIdx, internalNodesExceptCloth, device.d_objs, numParticles);
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

    void addOneLight(double3 position, double width, double height, double3 normal, double3 color, bool visible = false)
    {
        unsigned int prePtr, afterPtr;
        cudaMemcpy(&prePtr, device.d_objPtr, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        allocateLightOnDevice << < 1, 1 >> > (device.d_objs, device.d_objPtr, d_lightsIndex, device.d_lightPtr, position, width, height, normal, color, visible);
        checkCudaErrors(cudaDeviceSynchronize());
        cudaMemcpy(&afterPtr, device.d_objPtr, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        lightsCount++;
        allCount++;

        printf("Allocate Light [%d:%d) on device...\n", prePtr, afterPtr);
        objects.push_back({ "light", prePtr, afterPtr });
    }

    void addOneSphere(double3 position, double radius, double3 color)
    {
        unsigned int prePtr, afterPtr;
        cudaMemcpy(&prePtr, device.d_objPtr, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        allocateSphereOnDevice << < 1, 1 >> > (device.d_objs, device.d_objPtr, position, radius, color);
        checkCudaErrors(cudaDeviceSynchronize());
        cudaMemcpy(&afterPtr, device.d_objPtr, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        allCount++;

        printf("Allocate Sphere [%d:%d) on device...\n", prePtr, afterPtr);
        objects.push_back({ "sphere", prePtr, afterPtr });
    }

    void addFloor(double3 lt, double3 rt, double3 lb, double3 rb, double3 color)
    {
        unsigned int prePtr, afterPtr;
        cudaMemcpy(&prePtr, device.d_objPtr, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        allocateFloorOnDevice << < 1, 1 >> > (device.d_objs, device.d_objPtr, lt, rt, lb, rb, color);
        checkCudaErrors(cudaDeviceSynchronize());
        cudaMemcpy(&afterPtr, device.d_objPtr, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        allCount += 2;

        printf("Allocate Floor [%d:%d) on device...\n", prePtr, afterPtr);
        objects.push_back({ "floor", prePtr, afterPtr });
    }

    void addMeshes(const std::string& fileName, double3 position, double3 color, bool glass = false, const double scale = 1.0)
    {
        unsigned int prePtr, afterPtr;
        cudaMemcpy(&prePtr, device.d_objPtr, sizeof(unsigned int), cudaMemcpyDeviceToHost);

        Mesh mesh;
        mesh.loadFromFile(fileName, scale);
        mesh.transform(position);
        int num = mesh.triangles.size();
        MeshTriangle* d_triangles;
        cudaMalloc((void**)&d_triangles, num * sizeof(MeshTriangle));
        cudaMemcpy(d_triangles, mesh.triangles.data(), num * sizeof(MeshTriangle), cudaMemcpyHostToDevice);
        allocateMeshesOnDevice << < 1, 1 >> > (device.d_objs, device.d_objPtr, d_triangles, color, glass, num);
        checkCudaErrors(cudaDeviceSynchronize());
        cudaMemcpy(&afterPtr, device.d_objPtr, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        allCount += num;

        printf("Allocate Meshes [%d:%d) on device...\n", prePtr, afterPtr);
        objects.push_back({ "meshes", prePtr, afterPtr });
    }

    void addCloth()
    {
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
        
        unsigned int prePtr, afterPtr;
        cudaMemcpy(&prePtr, device.d_objPtr, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        allocateClothToDevice << < 1, 1 >> > (device.d_objs, device.d_objPtr, device.d_clothPtr, d_X, d_triIdx, division - 1);
        checkCudaErrors(cudaDeviceSynchronize());
        cudaMemcpy(&afterPtr, device.d_objPtr, sizeof(unsigned int), cudaMemcpyDeviceToHost);

        int clothTriCount = 2 * (division - 1) * (division - 1);
        allCount += clothTriCount;
        printf("Allocate Cloth [%d:%d) on device...\n", prePtr, afterPtr);
        objects.push_back({ "cloth", prePtr, afterPtr });
    }

};