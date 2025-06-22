#pragma once

#include <curand_kernel.h>
#include <vector>
#include <set>
#include <tuple>
#include <iterator>
#include <algorithm>

#include "rtmath.cuh"


const int division = 20;
const int numParticles = division * division;
const int numPartAxis = 3 * numParticles;

class Cloth
{
public:
    double width;  // cloth width
    double3 position;
    double simTime;
    double dt;

    double* X = new double[numPartAxis];  // positions
    double* V = new double[numPartAxis];  // velocities
    double* F = new double[numPartAxis];  // forces
    double* M = new double[numPartAxis * numPartAxis];    // mass field
    double* L = new double[numParticles * numParticles];  // Length
    int* edgeIdx = new int[(6 * division * division - 12 * division + 6) * 2];  // springs between particles
    int* triangleIdx = new int[6 * (division - 1) * (division - 1)];  // indices of particles forming triangles

public:
    Cloth(double _width, double3 _position) : width(_width), position(_position)
    {
        simTime = 0.0;
        dt = 0.001;
    }

public:
    void initialize() 
    {
        // init positions
        double dl = width / division;
        double dm = 0.01;
        for (int i = 0; i < numPartAxis; i++)
        {
            V[i] = 0.0;
            for (int j = 0; j < numPartAxis; j++)
            {
                M[i * numPartAxis + j] = 0.0;
            }
        }
        for (int i = 0; i < numParticles; i++)
        {
            X[i * 3 + 0] = -width / 2.0 + (i % division) * dl + position.x;
            X[i * 3 + 1] = position.y;
            X[i * 3 + 2] = width / 2.0 - (i / division) * dl + position.z;

            M[(i * 3 + 0) * numPartAxis + (i * 3 + 0)] = 1.0 / dm;
            M[(i * 3 + 1) * numPartAxis + (i * 3 + 1)] = 1.0 / dm;
            M[(i * 3 + 2) * numPartAxis + (i * 3 + 2)] = 1.0 / dm;
        }

        // init triangles
        for (int i = 0; i < division - 1; i++)
        {
            for (int j = 0; j < division - 1; j++)
            {
                int quad_id = 6 * (i * (division - 1) + j);
                triangleIdx[quad_id + 0] = i * division + j;
                triangleIdx[quad_id + 1] = (i + 1) * division + j;
                triangleIdx[quad_id + 2] = i * division + j + 1;

                triangleIdx[quad_id + 3] = i * division + j + 1;
                triangleIdx[quad_id + 4] = (i + 1) * division + j;
                triangleIdx[quad_id + 5] = (i + 1) * division + j + 1;
            }
        }

        // init edges
        using Triple = std::tuple<int, int, int>;  // (v0, v1, triangle index)
        std::vector<Triple> triple_list;
        triple_list.clear();
        for (int i = 0; i < 2 * (division - 1) * (division - 1); i++)
        {
            int a = triangleIdx[3 * i + 0];
            int b = triangleIdx[3 * i + 1];
            int c = triangleIdx[3 * i + 2];

            std::vector<std::pair<int, int>> edges = {
                {std::min(a, b), std::max(a, b)},
                {std::min(b, c), std::max(b, c)},
                {std::min(a, c), std::max(a, c)}
            };

            for (auto& e : edges) 
            {
                triple_list.emplace_back(e.first, e.second, i);
            }
        }
        std::sort(triple_list.begin(), triple_list.end());

        std::vector<std::pair<int, int>> edge_list;
        std::vector<std::pair<int, int>> neighbor_list;
        size_t i = 0;
        size_t l = triple_list.size();
        while (i < l) 
        {
            // 当前边
            int v0a = std::get<0>(triple_list[i]);
            int v1a = std::get<1>(triple_list[i]);
            int triA = std::get<2>(triple_list[i]);
            edge_list.emplace_back(v0a, v1a);

            // 查看是否有下一个边相同
            if (i + 1 < l) 
            {
                int v0b = std::get<0>(triple_list[i + 1]);
                int v1b = std::get<1>(triple_list[i + 1]);
                int triB = std::get<2>(triple_list[i + 1]);
                if (v0a == v0b && v1a == v1b)
                {
                    neighbor_list.emplace_back(triA, triB);
                    i += 2;
                    continue;
                }
            }
            i += 1;
        }

        for (const auto& tur : neighbor_list) 
        {
            int tri0 = tur.first;
            int tri1 = tur.second;
            std::set<int> tri0_set = {
                triangleIdx[3 * tri0 + 0],
                triangleIdx[3 * tri0 + 1],
                triangleIdx[3 * tri0 + 2]
            };
            std::set<int> tri1_set = {
                triangleIdx[3 * tri1 + 0],
                triangleIdx[3 * tri1 + 1],
                triangleIdx[3 * tri1 + 2]
            };
            std::vector<int> sym_diff;
            std::set_symmetric_difference(
                tri0_set.begin(), tri0_set.end(),
                tri1_set.begin(), tri1_set.end(),
                std::back_inserter(sym_diff)
            );
            if (sym_diff.size() == 2) 
            {
                edge_list.emplace_back(sym_diff[0], sym_diff[1]);
            }
            for (size_t i = 0; i < edge_list.size(); ++i) 
            {
                edgeIdx[2 * i + 0] = edge_list[i].first;
                edgeIdx[2 * i + 1] = edge_list[i].second;
            }
        }

        // init L
        for (size_t idx = 0; idx < 6 * division * division - 12 * division + 6; idx++)
        {
            int i = edgeIdx[2 * idx + 0];
            int j = edgeIdx[2 * idx + 1];
            double dx = X[3 * j + 0] - X[3 * i + 0];
            double dy = X[3 * j + 1] - X[3 * i + 1];
            double dz = X[3 * j + 2] - X[3 * i + 2];
            double restLen = sqrt(dx * dx + dy * dy + dz * dz);
            L[i * numParticles + j] = restLen;
        }

        //for (size_t i = 0; i < (6 * division * division - 12 * division + 6); i++)
        //{
        //    printf("(%d, %d)\n", edgeIdx[2 * i], edgeIdx[2 * i + 1]);
        //}
    }

    void Update()
    {
        simTime += dt;

        for (size_t i = 0; i < numParticles * 3; i++)
        {
            F[i] = 0.0;
        }
        for (size_t k = 0; k < 6 * division * division - 12 * division + 6; k++)
        {
            int i = edgeIdx[2 * k + 0];
            int e = edgeIdx[2 * k + 1];
            double3 x_i = make_double3(X[3 * i], X[3 * i + 1], X[3 * i + 2]);
            double3 x_e = make_double3(X[3 * e], X[3 * e + 1], X[3 * e + 2]);
            double3 x_ei = x_i - x_e;
            double ei_norm = Length(x_ei);
            double3 force = -1000.0 * (ei_norm - L[i * numParticles + e]) * x_ei / ei_norm;
            F[3 * i + 0] += force.x;
            F[3 * i + 1] += force.y;
            F[3 * i + 2] += force.z;
            F[3 * e + 0] += -force.x;
            F[3 * e + 1] += -force.y;
            F[3 * e + 2] += -force.z;
        }
        for (size_t i = 0; i < numParticles; i++)
            F[3 * i + 1] += -1.0 / numParticles * 9.8;
        // calculate velocity
        for (int i = 0; i < numParticles * 3; ++i)
        {
            for (int j = 0; j < numParticles * 3; ++j)
            {
                V[i] += dt * (M[i * numPartAxis + j] * F[j]);  // mass_field[i][j] @ f[j]
            }
        }

        // update positions
        for (int i = 0; i < numParticles; ++i) 
        {
            X[3 * i + 0] += dt * V[3 * i + 0];
            X[3 * i + 1] += dt * V[3 * i + 1];
            X[3 * i + 2] += dt * V[3 * i + 2];
        }
        // collision detection and response
        double r = 3.0;
        for (int i = 0; i < numParticles; i++) {
            double dx = X[i * 3 + 0];
            double dy = X[i * 3 + 1];
            double dz = X[i * 3 + 2];
            double dist2 = dx * dx + dy * dy + dz * dz;
            if (dist2 < r * r) {
                // 发生碰撞，进行位置修正
                double dist = sqrt(dist2);
                double penetration = r - dist;
                double nx = dx / dist;
                double ny = dy / dist;
                double nz = dz / dist;
                X[i * 3 + 0] = nx * r;
                X[i * 3 + 1] = ny * r;
                X[i * 3 + 2] = nz * r;
            }
        }
    }
};
