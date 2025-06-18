
#include <iostream>
#include "Window.cuh"

#include "Render.cuh"
#include "Hittable.cuh"
#include "Scene.cuh"


int main()
{
    const int nx = 800;
    const int ny = static_cast<int>(nx / 16.0 * 9.0);
    
    dim3 threads(8, 8);
    dim3 blocks((nx + 7) / 8, (ny + 7) / 8);
    
    Camera camera(nx, 16.0 / 9.0);
    Camera* d_camera;
    checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(Camera)));
    checkCudaErrors(cudaMemcpy(d_camera, &camera, sizeof(Camera), cudaMemcpyHostToDevice));

    Window app(nx, ny, &camera);
    unsigned char* d_cb;
    checkCudaErrors(cudaMalloc((void**)&d_cb, app.cb_size));

    Scene scene;
    scene.init();
    
    if (app.Init())
    {
        bool preStats = false;
        double preTime = 0;
        double t = 0;
        while (!app.Close())
        {
            if (app.PollInput())
                checkCudaErrors(cudaMemcpy(d_camera, &camera, sizeof(Camera), cudaMemcpyHostToDevice));
            if (!app.paused || app.paused != preStats)
            {
                t = (double)glfwGetTime();
                int sampleCount = app.sampleCount;
                if (app.paused)
                {
                    std::cout << "Rendering " << app.sampleCount << " sample count..." << std::endl;
                    t = preTime;
                }
                render <<< blocks, threads >>> (d_cb, d_camera, scene.d_lightsIndex, scene.device.d_objs, scene.internalNodes, scene.lightsCount, nx, ny, sampleCount, app.roughness, app.metallic, t);
                checkCudaErrors(cudaDeviceSynchronize());
                cudaMemcpy(app.img, d_cb, app.cb_size, cudaMemcpyDeviceToHost);
                scene.Update();
            }

            app.Update();
            preStats = app.paused;
            preTime = t;
        }
    }

    cudaFree(d_cb);
    cudaFree(d_camera);
}