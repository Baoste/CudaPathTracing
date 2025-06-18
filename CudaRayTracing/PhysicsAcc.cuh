#pragma once

#include "BVH.cuh"


// check for collision AABB
__host__ __device__ inline bool checkOverlap(const double3 p, const AABB& aabb)
{
    double radius = 0.01;
    return (p.x + radius >= aabb.min.x && p.x - radius <= aabb.max.x &&
        p.y + radius >= aabb.min.y && p.y - radius <= aabb.max.y &&
        p.z + radius >= aabb.min.z && p.z - radius <= aabb.max.z);
}

// calculate the distance from a point to a triangle
__host__ __device__ inline double3 pointTriangleDistance(const double3 p, const double3 a, const double3 b, const double3 c)
{
    double3 ab = b - a;
    double3 ac = c - a;
    double3 ap = p - a;

    double d1 = Dot(ab, ap);
    double d2 = Dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) return (p - a); // 顶点a

    double3 bp = p - b;
    double d3 = Dot(ab, bp);
    double d4 = Dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) return (p - b); // 顶点b

    double vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        double v = d1 / (d1 - d3);
        double3 q = a + ab * v;
        return (p - q); // 边ab
    }

    double3 cp = p - c;
    double d5 = Dot(ab, cp);
    double d6 = Dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) return (p - c); // 顶点c

    double vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        double w = d2 / (d2 - d6);
        double3 q = a + ac * w;
        return (p - q); // 边ac
    }

    double va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        double3 q = b + (c - b) * w;
        return (p - q); // 边bc
    }

    // 面内
    double denom = 1.0 / (va + vb + vc);
    double v = vb * denom;
    double w = vc * denom;
    double3 q = a + ab * v + ac * w;
    double3 offset = p - q;
    double3 normal = Unit(Cross(ab, ac));
    if (Dot(offset, normal) < 0.0)
        offset = -offset;
    return offset;
}

// iter tree for collision
__host__ __device__ inline double3 traverseIterative(const double3 p, Node* internalNodes, Hittable** objs, double epsilon)
{
    // Allocate traversal stack from thread-local memory,
    // and push NULL to indicate that there are no postponed nodes.
    double3 offset = make_double3(10.0, 10.0, 10.0);
    Node* stack[64];
    Node** stackPtr = stack;
    *stackPtr++ = NULL; // push

    bool isHit = false;

    // Traverse nodes starting from the root.
    Node* node = internalNodes;
    do
    {
        // Check each child node for overlap.
        Node* childL = node->childA;
        Node* childR = node->childB;
        bool overlapL = checkOverlap(p, childL->aabb);
        bool overlapR = checkOverlap(p, childR->aabb);

        // Query overlaps a leaf node => report collision.
        if (overlapL && childL->isLeaf)
        {
            Triangle tri = (*objs)[childL->objectID].triangle;
            double3 tmpOffset = pointTriangleDistance(p, tri.p0, tri.p1, tri.p2);
            //printf("%f %f %f\n", tmpOffset.x, tmpOffset.y, tmpOffset.z);
            if (SquaredLength(tmpOffset) < epsilon)
                offset = tmpOffset;
        }
        if (overlapR && childR->isLeaf)
        {
            Triangle tri = (*objs)[childR->objectID].triangle;
            double3 tmpOffset = pointTriangleDistance(p, tri.p0, tri.p1, tri.p2);
            if (SquaredLength(tmpOffset) < epsilon)
                offset = SquaredLength(offset) < SquaredLength(tmpOffset) ? offset : tmpOffset;
        }
        //printf("OFFSET: %f %f %f\n", offset.x, offset.y, offset.z);

        // Query overlaps an internal node => traverse.
        bool traverseL = (overlapL && !childL->isLeaf);
        bool traverseR = (overlapR && !childR->isLeaf);

        if (!traverseL && !traverseR)
            node = *--stackPtr; // pop
        else
        {
            node = (traverseL) ? childL : childR;
            if (traverseL && traverseR)
                *stackPtr++ = childR; // push
        }
    } while (node != NULL);

    return offset;
}

__host__ __device__ inline void collisionDetect(double* X, double* V, int idx, Node* internalNodes, Hittable** d_b)
{
    double epsilon = 0.01;
    double friction = 0.3;

    double3 p = make_double3(X[idx + 0], X[idx + 1], X[idx + 2]);
    double3 offset = traverseIterative(p, internalNodes, d_b, epsilon);

    if (Length(offset) < 1.0)
    {
        double3 n = Unit(offset);
        p = p - offset + n * epsilon;

        double3 v = make_double3(V[idx + 0], V[idx + 1], V[idx + 2]);
        double3 v_normal = Dot(v, n) * n;
        double3 v_tangent = v - v_normal;
        v = v_tangent * (1.0 - friction);
        v -= v_normal;

        X[idx + 0] = p.x;
        X[idx + 1] = p.y;
        X[idx + 2] = p.z;
        V[idx + 0] = 0.0;
        V[idx + 1] = 0.0;
        V[idx + 2] = 0.0;
    }
}

__global__ inline void PhysicsUpdate(double dt, double* X, double* V, double* F, double* M, double* L, int* edgeIdx,
    Node* internalNodes, Hittable** d_b, int numParticles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    int partIndex = idx * 3;
    // reset
    for (int i = 0; i < 3; ++i)
    {
        F[partIndex + i] = 0.0;
    }
    __syncthreads();
    // calculate forces
    for (size_t k = 0; k < 6 * division * division - 12 * division + 6; k++)
    {
        int i = edgeIdx[2 * k + 0];
        int e = edgeIdx[2 * k + 1];
        if (3 * i != partIndex && 3 * e != partIndex) continue; // only calculate forces for the current particle
        double3 x_i = make_double3(X[3 * i + 0], X[3 * i + 1], X[3 * i + 2]);
        double3 x_e = make_double3(X[3 * e + 0], X[3 * e + 1], X[3 * e + 2]);
        double3 x_ei = x_i - x_e;
        double ei_norm = Length(x_ei);
        //printf("Edge (%d, %d): (%f, %f, %f) -> (%f, %f, %f) = %f\n", i, e, x_i.x, x_i.y, x_i.z, x_e.x, x_e.y, x_e.z, ei_norm);
        double3 force = -100.0 * (ei_norm - L[i * numParticles + e]) * x_ei / ei_norm;
        if (3 * i == partIndex)
        {
            F[3 * i + 0] += force.x;
            F[3 * i + 1] += force.y;
            F[3 * i + 2] += force.z;
        }
        if (3 * e == partIndex)
        {
            F[3 * e + 0] += -force.x;
            F[3 * e + 1] += -force.y;
            F[3 * e + 2] += -force.z;
        }
    }
    __syncthreads();
    // gravity
    F[partIndex + 1] += -0.01 * 9.8;
    // drag force
    //double3 v = make_double3(V[partIndex + 0], V[partIndex + 1], V[partIndex + 2]);
    //double3 drag = -2.0 * Length(v) * v;
    //F[partIndex + 0] += drag.x;
    //F[partIndex + 1] += drag.y;
    //F[partIndex + 2] += drag.z;
    // calculate velocity
    for (int i = 0; i < numParticles * 3; i++)
    {
        V[partIndex + 0] += dt * (M[(partIndex + 0) * numPartAxis + i] * F[i]);
        V[partIndex + 1] += dt * (M[(partIndex + 1) * numPartAxis + i] * F[i]);
        V[partIndex + 2] += dt * (M[(partIndex + 2) * numPartAxis + i] * F[i]);
    }
    // check for collision
    //collisionDetect(X, V, partIndex, internalNodes, d_b);
    double friction = 0.3;
    double restitution = 0.0; // 反弹强度
    double r = 1.2;
    double3 center = make_double3(2.0, 1.0, 2.0);
    double3 p = make_double3(X[partIndex + 0], X[partIndex + 1], X[partIndex + 2]);
    double3 offset = p - center;
    double dist2 = SquaredLength(offset);
    if (dist2 < r * r) 
    {
        double dist = sqrt(dist2);
        double penetration = r - dist;
        double3 n = offset / dist;

        p = center + n * r;
        X[partIndex + 0] = p.x;
        X[partIndex + 1] = p.y;
        X[partIndex + 2] = p.z;

        double3 v = make_double3(V[partIndex + 0], V[partIndex + 1], V[partIndex + 2]);
        double3 v_normal = Dot(v, n) * n;
        double3 v_tangent = v - v_normal;
        v = v_tangent * (1.0 - friction) - restitution * v_normal;

        V[partIndex + 0] = v.x;
        V[partIndex + 1] = v.y;
        V[partIndex + 2] = v.z;
    }
    
    // update position
    for (int i = 0; i < 3; i++)
    {
        X[partIndex + i] += dt * V[partIndex + i];
    }
}