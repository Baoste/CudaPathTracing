#pragma once

#include <windows.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>

#include <cuda_runtime.h>

struct CameraInfo
{
    double3 background;
    double width;
    double3 lookFrom;
    double3 lookAt;
    double vFov;
    CameraInfo() : background{ 0.0, 0.0, 0.0 }, width(0.0), lookFrom{ 0.0, 0.0, 1.0 }, lookAt{ 0.0, 0.0, 0.0 }, vFov(90.0) {}
};

struct LightInfo
{
    double3 center;
    double width, height;
    double3 normal;
    double3 color;
    LightInfo() : center{ 0.0, 0.0, 0.0 }, width(0.0), height(0.0), normal{ 0.0, 0.0, 1.0 }, color{ 1.0, 1.0, 1.0 } {}
};

struct SphereInfo
{
    double3 center;
    double radius;
    double3 color;
    double alphaX, alphaY;
    SphereInfo() : center{ 0.0, 0.0, 0.0 }, radius(0.0), color{ 1.0, 1.0, 1.0 }, alphaX(0.5), alphaY(0.5) {}
};

struct FloorInfo
{
    double3 lt, rt, lb, rb;
    double3 color;
    double alphaX, alphaY;
    FloorInfo() : lt{ 0.0, 0.0, 0.0 }, rt{ 0.0, 0.0, 0.0 }, lb{ 0.0, 0.0, 0.0 }, rb{ 0.0, 0.0, 0.0 }, color{ 1.0, 1.0, 1.0 }, alphaX(0.5), alphaY(0.5) {}
};

struct MeshInfo
{
    std::string path;
    double3 center;
    double3 color;
    double alphaX, alphaY;
    bool glass;
    double scale; 
    MeshInfo() : path(""), center{ 0.0, 0.0, 0.0 }, color{ 1.0, 1.0, 1.0 }, alphaX(0.5), alphaY(0.5), glass(false), scale(1.0) {}
};

struct ClothInfo
{
    double3 center;
    double width;
    double3 color;
    ClothInfo() : center{ 0.0, 0.0, 0.0 }, width(0.0), color{ 1.0, 1.0, 1.0 } {}
};

class IniParser
{
public:
    void Parse(const std::string& filename);

    CameraInfo camera;
    std::vector<LightInfo> lights;
    std::vector<SphereInfo> spheres;
    std::vector<FloorInfo> floors;
    std::vector<MeshInfo> meshes;
    bool hasCloth = false;
    ClothInfo cloth;

private:
    double3 parseVec3(const std::string& str);
};

