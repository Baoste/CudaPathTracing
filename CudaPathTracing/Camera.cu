
#include "Camera.cuh"
#include "stb_image.h"

Camera::Camera(int _width, double _aspectRatio, Color _background, double3 _lookFrom, double3 _lookAt, double _vFov, const std::string& skyBox)
    : width(_width), aspectRatio(_aspectRatio), background(_background), lookFrom(_lookFrom), lookAt(_lookAt), vFov(_vFov)
{
    isMoving = false;
    preLookFrom = _lookFrom;
    preLookAt = _lookAt;

    height = width / aspectRatio < 1 ? 1 : width / aspectRatio;

    double focalLength = Length(lookFrom - lookAt);
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

    d_image = NULL;
    if (!skyBox.empty())
    {
        unsigned char* image = stbi_load(skyBox.c_str(), &imgWidth, &imgHeight, &imgChannels, 0);
        size_t img_size = imgWidth * imgHeight * imgChannels * sizeof(unsigned char);
        cudaMalloc(&d_image, img_size);
        cudaMemcpy(d_image, image, img_size, cudaMemcpyHostToDevice);
    }
}
