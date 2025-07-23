#pragma once

#include <vector>
#include <string>

#include "Device.cuh"
#include "Cloth.cuh"
#include "PhysicsAcc.cuh"
#include "IniParser.h"
#include "stb_image.h"

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

    double3 minBoundary;
    double3 maxBoundary;

private:
    uint64_t* d_mortons;
    unsigned int* d_objIdx;

    uint64_t* d_mortonsExceptCloth;
    unsigned int* d_objIdxExceptCloth;
    double* d_X, * d_V, * d_F, * d_M, * d_L;
    int* d_edgeIdx, * d_triIdx;

    Cloth cloth;

public:
    Scene() : cloth(3.0, make_double3(0.0, 2.3, 0.0))
    {
        allCount = 0;
        lightsCount = 0;
        cudaMalloc((void**)&d_camera, sizeof(Camera));
        cudaMalloc((void**)&d_selectPtr, sizeof(int));

        minBoundary = make_double3(INF, INF, INF);
        maxBoundary = make_double3(-INF, -INF, -INF);
    }
    ~Scene()
    {
        cudaFree(d_mortons);
        cudaFree(d_objIdx);
        cudaFree(internalNodes);
        cudaFree(leafNodes);
        cudaFree(d_camera);
    }

    void init(const IniParser& parser)
    {
        // light
        cudaMalloc((void**)&d_lightsIndex, 10 * sizeof(unsigned int));
        for (auto light : parser.lights)
            addOneLight(light.center, light.width, light.height, light.normal, light.color, light.visible);

        // sphere
        for (auto sphere : parser.spheres)
            addOneSphere(sphere.center, sphere.radius, sphere.color, sphere.alphaX, sphere.alphaY, sphere.type);

        // floor
        for (auto floor : parser.floors)
            addFloor(floor.lt, floor.rt, floor.lb, floor.rb, floor.color, floor.alphaX, floor.alphaY, floor.type);

        // mesh
        for (auto mesh : parser.meshes)
            addMeshes(mesh.path, mesh.texture, mesh.center, mesh.rotation, mesh.scale, mesh.color, mesh.alphaX, mesh.alphaY, mesh.type);

        // cloth
        if (parser.hasCloth)
        {
            cudaMalloc((void**)&d_mortonsExceptCloth, allCount * sizeof(uint64_t));
            cudaMalloc((void**)&d_objIdxExceptCloth, allCount * sizeof(unsigned int));
            cudaMalloc((void**)&leafNodesExceptCloth, allCount * sizeof(Node));
            cudaMalloc((void**)&internalNodesExceptCloth, (allCount - 1) * sizeof(Node));
            device.buildBVH(device.d_objs, leafNodesExceptCloth, internalNodesExceptCloth, d_mortonsExceptCloth, d_objIdxExceptCloth, allCount, minBoundary, maxBoundary);
            checkCudaErrors(cudaDeviceSynchronize());

            // allocate cloth triangles to device
            addCloth();
            cudaMalloc((void**)&d_mortons, allCount * sizeof(uint64_t));
            cudaMalloc((void**)&d_objIdx, allCount * sizeof(unsigned int));
            cudaMalloc((void**)&internalNodes, (allCount - 1) * sizeof(Node));
            cudaMalloc((void**)&leafNodes, allCount * sizeof(Node));

            double simTime = 1.6;
            printf("Start sim physics for %f seconds...\n", simTime);
            while (cloth.simTime < simTime)
            {
                Update();
                break;
            }
        }
        else
        {
            cudaMalloc((void**)&d_mortons, allCount * sizeof(uint64_t));
            cudaMalloc((void**)&d_objIdx, allCount * sizeof(unsigned int));
            cudaMalloc((void**)&internalNodes, (allCount - 1) * sizeof(Node));
            cudaMalloc((void**)&leafNodes, allCount * sizeof(Node));

            device.buildBVH(device.d_objs, leafNodes, internalNodes, d_mortons, d_objIdx, allCount, minBoundary, maxBoundary);
            checkCudaErrors(cudaDeviceSynchronize());
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

        device.buildBVH(device.d_objs, leafNodes, internalNodes, d_mortons, d_objIdx, allCount, minBoundary, maxBoundary);
        checkCudaErrors(cudaDeviceSynchronize());

    }

    void addOneLight(double3 position, double width, double height, double3 normal, double3 color, bool visible)
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

    void addOneSphere(double3 position, double radius, double3 color, double alphaX, double alphaY, MaterialType type = MaterialType::M_OPAQUE)
    {
        unsigned int prePtr, afterPtr;
        cudaMemcpy(&prePtr, device.d_objPtr, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        allocateSphereOnDevice << < 1, 1 >> > (device.d_objs, device.d_objPtr, position, radius, color, alphaX, alphaY, type);
        checkCudaErrors(cudaDeviceSynchronize());
        cudaMemcpy(&afterPtr, device.d_objPtr, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        allCount++;

        printf("Allocate Sphere [%d:%d) on device...\n", prePtr, afterPtr);
        objects.push_back({ "sphere", prePtr, afterPtr });
    }

    void addFloor(double3 lt, double3 rt, double3 lb, double3 rb, double3 color, double alphaX, double alphaY, MaterialType type = MaterialType::M_OPAQUE)
    {
        unsigned int prePtr, afterPtr;
        cudaMemcpy(&prePtr, device.d_objPtr, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        allocateFloorOnDevice << < 1, 1 >> > (device.d_objs, device.d_objPtr, lt, rt, lb, rb, color, alphaX, alphaY, type);
        checkCudaErrors(cudaDeviceSynchronize());
        cudaMemcpy(&afterPtr, device.d_objPtr, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        allCount += 2;

        printf("Allocate Floor [%d:%d) on device...\n", prePtr, afterPtr);
        objects.push_back({ "floor", prePtr, afterPtr });
    }

    //void addBzeier(const double3 p0, const double3 p1, const double3 p2, const double3 p3,
    //    const double3 p4, const double3 p5, const double3 p6, const double3 p7,
    //    const double3 p8, const double3 p9, const double3 p10, const double3 p11,
    //    const double3 p12, const double3 p13, const double3 p14, const double3 p15,
    //    const double3 color, double alphaX, double alphaY)
    //{
    //    unsigned int prePtr, afterPtr;
    //    cudaMemcpy(&prePtr, device.d_objPtr, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    //    allocateBzeierOnDevice << < 1, 1 >> > (device.d_objs, device.d_objPtr, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, color, alphaX, alphaY);
    //    checkCudaErrors(cudaDeviceSynchronize());
    //    cudaMemcpy(&afterPtr, device.d_objPtr, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    //    allCount ++;

    //    printf("Allocate Floor [%d:%d) on device...\n", prePtr, afterPtr);
    //    objects.push_back({ "floor", prePtr, afterPtr });
    //}

    void addMeshes(const std::string& fileName, const std::string& texture, double3 position, double rotation, double scale, double3 color, double alphaX, double alphaY, MaterialType type = MaterialType::M_OPAQUE)
    {
        unsigned int prePtr, afterPtr;
        cudaMemcpy(&prePtr, device.d_objPtr, sizeof(unsigned int), cudaMemcpyDeviceToHost);

        Mesh mesh;
        mesh.loadFromFile(fileName, minBoundary, maxBoundary, scale, rotation);
        mesh.transform(position);
        int num = mesh.triangles.size();

        // load texture img
        int width, height, channels;
        unsigned char* image = stbi_load(texture.c_str(), &width, &height, &channels, 0);
        unsigned char* d_image = NULL;
        if (image) 
        {
            size_t img_size = width * height * channels * sizeof(unsigned char);
            cudaMalloc(&d_image, img_size);
            cudaMemcpy(d_image, image, img_size, cudaMemcpyHostToDevice);
        }

        MeshTriangle* d_triangles;
        MeshUV* d_uvs;
        MeshVn* d_vns;
        cudaMalloc((void**)&d_triangles, num * sizeof(MeshTriangle));
        cudaMalloc((void**)&d_uvs, num * sizeof(MeshUV));
        cudaMalloc((void**)&d_vns, num * sizeof(MeshVn));
        cudaMemcpy(d_triangles, mesh.triangles.data(), num * sizeof(MeshTriangle), cudaMemcpyHostToDevice);
        cudaMemcpy(d_uvs, mesh.uvs.data(), num * sizeof(MeshUV), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vns, mesh.vns.data(), num * sizeof(MeshVn), cudaMemcpyHostToDevice);
        allocateMeshesOnDevice << < 1, 1 >> > (device.d_objs, device.d_objPtr, d_triangles, d_image, width, height, channels, d_uvs, d_vns, color, alphaX, alphaY, type, num);
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