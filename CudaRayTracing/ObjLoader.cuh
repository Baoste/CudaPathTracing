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

class Mesh
{
public:
    std::vector<MeshTriangle> triangles;
    Mesh() = default;

public:
    // Load mesh from a file
    bool loadFromFile(const std::string& filename) 
    {
        triangles.clear();
        loadMesh(filename, triangles);
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
    void loadMesh(const std::string& filename, std::vector<MeshTriangle>& triangles)
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

        for (const auto& shape : shapes) 
        {
            size_t index_offset = 0;
            for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++) {
                size_t fv = shape.mesh.num_face_vertices[f];
                if (fv != 3) continue;

                MeshTriangle tri;
                for (size_t v = 0; v < 3; v++) {
                    tinyobj::index_t idx = shape.mesh.indices[index_offset + v];
                    int vidx = idx.vertex_index;

                    double x = attrib.vertices[3 * vidx + 0];
                    double y = attrib.vertices[3 * vidx + 1];
                    double z = attrib.vertices[3 * vidx + 2];
                    double3 p = make_double3(x, y, z);

                    if (v == 0) tri.p0 = p;
                    else if (v == 1) tri.p1 = p;
                    else if (v == 2) tri.p2 = p;
                }
                triangles.push_back(tri);
                index_offset += fv;
            }
        }
    }
};
