
#include "Camera.cuh"

Camera::Camera(int _width, double _aspectRatio, Color _background, double3 _lookFrom, double3 _lookAt, double _vFov)
    : width(_width), aspectRatio(_aspectRatio), background(_background), lookFrom(_lookFrom), lookAt(_lookAt), vFov(_vFov)
{
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
}
