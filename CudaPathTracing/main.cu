
#include "Window.cuh"
#include "Render.cuh"
#include "Hittable.cuh"
#include "IniParser.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

int main()
{
    std::string sceneINIPath;
    std::cout << "Enter scene ini file path (default: scene0.ini):" << std::endl;
    std::getline(std::cin, sceneINIPath);
    // default
    if (sceneINIPath.empty()) 
        sceneINIPath = "0";

    sceneINIPath = "C:/Users/59409/source/repos/CudaPathTracing/CudaPathTracing/scene_ini/Scene" + sceneINIPath + ".ini";
    IniParser parser;
    parser.Parse(sceneINIPath);

    int nx = parser.camera.width;
    int ny = static_cast<int>(nx / parser.camera.aspectRatio);
    
    const int threadsNum = 16;
    dim3 threads(threadsNum, threadsNum);
    dim3 blocks((nx + threadsNum - 1) / threadsNum, (ny + threadsNum - 1) / threadsNum);
    
    Scene scene;
    scene.init(parser);

    Camera camera(nx, parser.camera.aspectRatio, parser.camera.background, parser.camera.lookFrom, parser.camera.lookAt, parser.camera.vFov, parser.camera.skyBox);
    checkCudaErrors(cudaMemcpy(scene.d_camera, &camera, sizeof(Camera), cudaMemcpyHostToDevice));


    Window app(nx, ny, &camera, &scene);
    
    double3* d_pic, * d_picPrevious, * d_picBeforeGussian;
    cudaMalloc((void**)&d_pic, nx * ny * sizeof(double3));
    cudaMalloc((void**)&d_picPrevious, nx * ny * sizeof(double3));
    cudaMalloc((void**)&d_picBeforeGussian, nx * ny * sizeof(double3));

    double4* d_gBuffer;
    cudaMalloc((void**)&d_gBuffer, nx * ny * sizeof(double4));
    double3* d_gBufferPosition;
    cudaMalloc((void**)&d_gBufferPosition, nx * ny * sizeof(double3));

    if (app.Init())
    {
        bool preStats = false;
        double preTime, nowTime = 0;

        clear <<< blocks, threads >>> (d_pic, nx, ny);
        cudaDeviceSynchronize();

        while (!app.Close())
        {
            preTime = nowTime;
            nowTime = (double)glfwGetTime();
            app.deltaTime = nowTime - preTime;

            if (app.PollInput())
            {
                cudaMemcpy(scene.d_camera, &camera, sizeof(Camera), cudaMemcpyHostToDevice);
                //cudaDeviceSynchronize();
            }
            if (!app.paused || app.paused != preStats)
            {
                int sampleCount = app.sampleCount;
                if (app.paused)
                {
                    std::cout << "Rendering " << app.sampleCount << " sample count..." << std::endl;
                    preStats = app.paused;
                    render <<< blocks, threads >>> (app.devicePtr, d_pic, d_picPrevious, d_picBeforeGussian, d_gBuffer, d_gBufferPosition,
                        scene.d_camera, scene.d_lightsIndex, scene.device.d_objs, scene.internalNodes, scene.lightsCount, 
                        nx, ny, sampleCount, nowTime, RenderType::STATIC);
                    continue;
                }

                switch (app.currentType)
                {
                case 0:
                    render << < blocks, threads >> > (app.devicePtr, d_pic, d_picPrevious, d_picBeforeGussian, d_gBuffer, d_gBufferPosition,
                        scene.d_camera, scene.d_lightsIndex, scene.device.d_objs, scene.internalNodes, scene.lightsCount,
                        nx, ny, sampleCount, nowTime, RenderType::REAL_TIME_NOT_SAMPLE);
                    cudaDeviceSynchronize();
                    pic2RGBW << < blocks, threads >> > (app.devicePtr, d_pic, nx, ny);
                    cudaDeviceSynchronize();
                    break;
                case 1:
                    render << < blocks, threads >> > (app.devicePtr, d_pic, d_picPrevious, d_picBeforeGussian, d_gBuffer, d_gBufferPosition,
                        scene.d_camera, scene.d_lightsIndex, scene.device.d_objs, scene.internalNodes, scene.lightsCount,
                        nx, ny, sampleCount, nowTime, RenderType::REAL_TIME);
                    cudaDeviceSynchronize();
                    
                    gaussianSeparate <<< blocks, threads >>> (d_pic, d_picBeforeGussian, d_gBuffer, app.sigmaG, app.sigmaR, app.sigmaN, app.sigmaD, nx, ny, true);
                    cudaDeviceSynchronize();
                    copyPic << < blocks, threads >> > (d_picBeforeGussian, d_pic, nx, ny);
                    cudaDeviceSynchronize();
                    gaussianSeparate <<< blocks, threads >>> (d_pic, d_picBeforeGussian, d_gBuffer, app.sigmaG, app.sigmaR, app.sigmaN, app.sigmaD, nx, ny, false);
                    cudaDeviceSynchronize();
                    //gaussian << < blocks, threads >> > (d_pic, d_picBeforeGussian, d_gBuffer, app.sigmaG, app.sigmaR, app.sigmaN, app.sigmaD, nx, ny);
                    //cudaDeviceSynchronize();
                    
                    addPrevious << < blocks, threads >> > (d_pic, d_picPrevious, d_gBufferPosition, scene.d_camera, nx, ny);
                    cudaDeviceSynchronize();
                    camera.isMoving = false;
                    cudaMemcpy(scene.d_camera, &camera, sizeof(Camera), cudaMemcpyHostToDevice);

                    pic2RGBW << < blocks, threads >> > (app.devicePtr, d_pic, nx, ny);
                    cudaDeviceSynchronize();
                    break;
                case 2:
                    render << < blocks, threads >> > (app.devicePtr, d_pic, d_picPrevious, d_picBeforeGussian, d_gBuffer, d_gBufferPosition,
                        scene.d_camera, scene.d_lightsIndex, scene.device.d_objs, scene.internalNodes, scene.lightsCount,
                        nx, ny, sampleCount, nowTime, RenderType::NORMAL);
                    cudaDeviceSynchronize();
                    break;
                case 3:
                    render << < blocks, threads >> > (app.devicePtr, d_pic, d_picPrevious, d_picBeforeGussian, d_gBuffer, d_gBufferPosition,
                        scene.d_camera, scene.d_lightsIndex, scene.device.d_objs, scene.internalNodes, scene.lightsCount,
                        nx, ny, sampleCount, nowTime, RenderType::DEPTH);
                    cudaDeviceSynchronize();
                    break;
                }

                
                if (parser.hasCloth)
                    scene.Update();
            }

            app.Update();
            preStats = app.paused;
        }
    }
}