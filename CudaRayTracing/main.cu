
#include "Window.cuh"
#include "Render.cuh"
#include "Hittable.cuh"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

int main()
{
    const int nx = 800;
    const int ny = static_cast<int>(nx / 16.0 * 9.0);
    
    dim3 threads(8, 8);
    dim3 blocks((nx + 7) / 8, (ny + 7) / 8);
    
    Scene scene;
    scene.init();

    Camera camera(nx, 16.0 / 9.0);
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
                // scene.Update();
            }

            app.Update();
            preStats = app.paused;
            preTime = t;
        }
    }
}