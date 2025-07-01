#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "tiny_obj_loader.h"
#include <iostream>

struct MeshTriangle 
{
    double3 p0;
    double3 p1;
    double3 p2;
};

struct MeshUV
{
    double2 p0;
    double2 p1;
    double2 p2;
};

class Mesh
{
public:
    std::vector<MeshTriangle> triangles;
    std::vector<MeshUV> uvs;
    Mesh() = default;

public:
    // Load mesh from a file
    bool loadFromFile(const std::string& filename, const double scale, const double rotation = -30.0)
    {
        triangles.clear();
        loadMesh(filename, triangles, scale, rotation);
        return !triangles.empty();
    }

    void transform(double3 position)
    {
        for (auto& triangle : triangles)
        {
            triangle.p0 += position;
            triangle.p1 += position;
            triangle.p2 += position;
        }
    }

private:
    void loadMesh(const std::string& filename, std::vector<MeshTriangle>& triangles, const double scale, const double rotation)
    {
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;

        std::string warn, err;

        bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filename.c_str());
        if (!ret)
        {
            std::cerr << err << std::endl;
            return;
        }

        double theta = DegreesToRadians(rotation);
        double cos_theta = cos(theta);
        double sin_theta = sin(theta);
        double3 rotation_x = make_double3(cos_theta, 0.0, sin_theta);
        double3 rotation_y = make_double3(0.0, 1.0, 0.0);
        double3 rotation_z = make_double3(-sin_theta, 0.0, cos_theta);

        for (const auto& shape : shapes) 
        {
            size_t index_offset = 0;
            for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++) {
                size_t fv = shape.mesh.num_face_vertices[f];
                if (fv != 3) continue;

                MeshTriangle tri;
                MeshUV muv;

                for (size_t v = 0; v < 3; v++) 
                {
                    tinyobj::index_t idx = shape.mesh.indices[index_offset + v];
                    int vidx = idx.vertex_index;

                    double x = attrib.vertices[3 * vidx + 0];
                    double y = attrib.vertices[3 * vidx + 1];
                    double z = attrib.vertices[3 * vidx + 2];
                    double3 p = make_double3(x, y, z);
                    
                    // rotate and scale
                    x = Dot(p, rotation_x);
                    y = Dot(p, rotation_y);
                    z = Dot(p, rotation_z);
                    p = make_double3(x, y, z) * scale;

                    if (v == 0) tri.p0 = p;
                    else if (v == 1) tri.p1 = p;
                    else if (v == 2) tri.p2 = p;
                    
                    // UV
                    int tidx = idx.texcoord_index;
                    if (tidx >= 0)
                    {
                        double uu = attrib.texcoords[2 * tidx + 0];
                        double vv = attrib.texcoords[2 * tidx + 1];
                        double2 uv_p = make_double2(uu, vv);
                        if (v == 0) muv.p0 = uv_p;
                        else if (v == 1) muv.p1 = uv_p;
                        else if (v == 2) muv.p2 = uv_p;
                    }
                    else
                    {
                        if (v == 0) muv.p0 = make_double2(0.0, 0.0);
                        else if (v == 1) muv.p1 = make_double2(0.0, 0.0);
                        else if (v == 2) muv.p2 = make_double2(0.0, 0.0);
                    }
                }

                triangles.push_back(tri);
                uvs.push_back(muv);

                index_offset += fv;
            }
        }
    }
};
