
#include <iostream>
#include "Window.h"

#include "Render.cuh"
#include "Hittable.cuh"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line)
{
    if (result)
    {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}


__global__ void allocateOnDevice(Hittable** d_b, const size_t size)
{
    *d_b = (Hittable*)malloc(size * sizeof(Hittable));  // raw allocation

    // ÏÔÊ½ placement new
    new (&(*d_b)[0]) Hittable(Sphere(make_double3(0, 0, 0), 1.0));
    new (&(*d_b)[1]) Hittable(Sphere(make_double3(0, -11, 0), 10.0));
}

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

    Hittable** d_objs;
    size_t objsCount = 2;
    cudaMalloc((void**)&d_objs, sizeof(Hittable**));
    allocateOnDevice <<< 1, 1 >>> (d_objs, objsCount);

    if (app.Init())
    {
        while (!app.Close())
        {
            app.PollInput();
            if (!app.paused)
            {
                double t = (double)glfwGetTime();
                render <<< blocks, threads >>> (d_cb, d_camera, d_objs, objsCount, nx, ny, t);
                //checkCudaErrors(cudaDeviceSynchronize());
                checkCudaErrors(cudaMemcpy(app.img, d_cb, app.cb_size, cudaMemcpyDeviceToHost));
            }

            app.Update();
        }
    }

    cudaFree(d_cb);
    cudaFree(d_camera);
}