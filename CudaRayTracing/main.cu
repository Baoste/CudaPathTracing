
#include <iostream>
#include "Window.h"

#include "Render.cuh"
#include "Hittable.cuh"
#include "Scene.cuh"


int main()
{
    const int nx = 800;
    const int ny = static_cast<int>(nx / 16.0 * 9.0);
    
    dim3 threads(8, 8);
    dim3 blocks((nx + 7) / 8, (ny + 7) / 8);

    Window app(nx, ny);
    unsigned char* d_cb;
    checkCudaErrors(cudaMalloc((void**)&d_cb, app.cb_size));
    
    Camera camera(nx, 16.0 / 9.0);
    Camera* d_camera;
    checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(Camera)));
    checkCudaErrors(cudaMemcpy(d_camera, &camera, sizeof(Camera), cudaMemcpyHostToDevice));

    Scene scene;
    scene.init();

    if (app.Init())
    {
        bool preStats = false;
        double preTime = 0;
        double t = 0;
        while (!app.Close())
        {
            app.PollInput();
            if (!app.paused || app.paused != preStats)
            {
                t = (double)glfwGetTime();
                int sampleCount = app.sampleCount;
                if (app.paused)
                {
                    std::cout << "Rendering " << app.sampleCount << " sample count..." << std::endl;
                    t = preTime;
                }
                render <<< blocks, threads >>> (d_cb, d_camera, scene.d_lightsIndex, scene.d_objs, scene.internalNodes, scene.lightsCount, nx, ny, sampleCount, t);
                // checkCudaErrors(cudaDeviceSynchronize());
                cudaMemcpy(app.img, d_cb, app.cb_size, cudaMemcpyDeviceToHost);
            }

            app.Update();
            preStats = app.paused;
            preTime = t;
        }
    }

    cudaFree(d_cb);
    cudaFree(d_camera);
}