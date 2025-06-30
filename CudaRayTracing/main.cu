
#include "Window.cuh"
#include "Render.cuh"
#include "Hittable.cuh"
#include "IniParser.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

int main()
{
    IniParser parser;
    parser.Parse("C:/Users/59409/source/repos/CudaRayTracing/CudaRayTracing/scene_ini/Scene3.ini");

    int nx = parser.camera.width;
    int ny = static_cast<int>(nx / 16.0 * 9.0);
    
    const int threadsNum = 16;
    dim3 threads(threadsNum, threadsNum);
    dim3 blocks((nx + threadsNum - 1) / threadsNum, (ny + threadsNum - 1) / threadsNum);
    
    Scene scene;
    scene.init(parser);

    Camera camera(nx, 16.0 / 9.0, parser.camera.background, parser.camera.lookFrom, parser.camera.lookAt, parser.camera.vFov);
    checkCudaErrors(cudaMemcpy(scene.d_camera, &camera, sizeof(Camera), cudaMemcpyHostToDevice));


    Window app(nx, ny, &camera, &scene);

    if (app.Init())
    {
        bool preStats = false;
        double preTime = 0;
        double t = 0;
        while (!app.Close())
        {
            if (app.PollInput())
                checkCudaErrors(cudaMemcpy(scene.d_camera, &camera, sizeof(Camera), cudaMemcpyHostToDevice));
            if (!app.paused || app.paused != preStats)
            {
                t = (double)glfwGetTime();
                int sampleCount = app.sampleCount;
                if (app.paused)
                {
                    std::cout << "Rendering " << app.sampleCount << " sample count..." << std::endl;
                    t = preTime;
                }
                render <<< blocks, threads >>> (app.devicePtr, scene.d_camera, scene.d_lightsIndex, scene.device.d_objs, scene.internalNodes, scene.lightsCount, nx, ny, sampleCount, t);
                checkCudaErrors(cudaDeviceSynchronize());
                
                if (parser.hasCloth)
                    scene.Update();
            }

            app.Update();
            preStats = app.paused;
            preTime = t;
        }
    }
}