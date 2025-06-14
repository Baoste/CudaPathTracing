#pragma once

#include "Ray.cuh"

class Camera
{
public:
    Camera(int _width, double _aspectRatio);
    Color background = make_double3(0.5, 0.7, 1.0);
    //Color background = make_double3(0.0, 0.0, 0.0);
    double aspectRatio = 16.0 / 9.0;
    int width = 640;
    int height = 1;
    int samplesPerPixel = 20;
    int maxDepth = 20;

    double3 lookFrom = make_double3(0, 0, 2);
    double3 lookAt = make_double3(0, 0, -1);
    double3 vUp = make_double3(0, 1, 0);
    // angel
    double vFov = 90.0;

    double3 u, v, w;

public:
    __device__ inline Ray getRandomSampleRay(const int& x, const int& y, const double& randx, const double& randy) const {
        double3 pixel = pixel_center +
            (randx - 0.5 + x) * delta_h +
            (randy - 0.5 + y) * delta_v;
        return Ray(lookFrom, pixel - lookFrom, 0.0);
    }

private:
    double3 delta_h;
    double3 delta_v;
    double3 pixel_center;
};

