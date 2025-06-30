#pragma once

#include "Ray.cuh"
#include <iostream>

class Camera
{
public:
    Camera(int _width, double _aspectRatio, Color _background, double3 _lookFrom, double3 _lookAt, double _vFov);

    Color background;
    double aspectRatio;
    int width;
    int height = 1;
    int samplesPerPixel = 1;
    int maxDepth = 1;

    double3 lookFrom;
    double3 lookAt;
    double3 vUp = make_double3(0, 1, 0);
    // angel
    double vFov;

    double3 u, v, w;

public:
    __device__ inline Ray getRandomSampleRay(const int x, const int y, const double randx, const double randy) const 
    {
        double3 pixel = pixel_center +
            (randx - 0.5 + x) * delta_h +
            (randy - 0.5 + y) * delta_v;
        return Ray(lookFrom, pixel - lookFrom, 0.0);
    }
    __device__ inline Ray getSampleRay(const int x, const int y) const 
    {
        double3 pixel = pixel_center + x * delta_h + y * delta_v;
        return Ray(lookFrom, pixel - lookFrom, 0.0);
    }
    __host__ __device__ inline void move(double p, double t, double x)
    {
        // TODO need to change axies
        double3 offset = lookFrom - lookAt;
        double focalLength = Length(offset);
        double updateT = acos(offset.y / focalLength);
        if (focalLength > 5.0 || x > 0.0)
            focalLength += x;
        if (updateT < PI - 0.1 && t < 0.0 || updateT > 0.1 && t > 0.0)
            updateT -= t;
        double updateP = atan2(offset.z, offset.x) - p;
        lookFrom = lookAt + make_double3(
            focalLength * sin(updateT) * cos(updateP),
            focalLength * cos(updateT),
            focalLength * sin(updateT) * sin(updateP)
        );

        double theta = DegreesToRadians(vFov);
        double h = tan(theta / 2);
        double viewportHeight = 2.0 * h * focalLength;
        double viewportWidth = static_cast<double>(width) / height * viewportHeight;

        w = Unit(lookFrom - lookAt);
        u = Unit(Cross(vUp, w));
        v = Cross(w, u);

        double3 horizontal = viewportWidth * u;
        double3 vertical = viewportHeight * v;
        delta_h = horizontal / width;
        delta_v = vertical / height;
        // upper left corner of the image
        double3 _p_ul = lookFrom - horizontal / 2 - vertical / 2 - focalLength * w;
        pixel_center = _p_ul + delta_h / 2 + delta_v / 2;
    }

private:
    double3 delta_h;
    double3 delta_v;
    double3 pixel_center;
};
