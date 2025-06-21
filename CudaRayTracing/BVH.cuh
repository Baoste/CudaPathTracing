#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include "device_launch_parameters.h"
#include "Hittable.cuh"


// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
__device__ inline unsigned int expandBits(unsigned int v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
__device__ inline unsigned int morton3D(double3 v)
{
    // normalize
    double3 sceneMin = make_double3(-4.0, -1.0, -4.0);
    double3 sceneMax = make_double3(4.0, 4.0, 4.0);
    double3 normalized;

    normalized.x = (v.x - sceneMin.x) / (sceneMax.x - sceneMin.x);
    normalized.y = (v.y - sceneMin.y) / (sceneMax.y - sceneMin.y);
    normalized.z = (v.z - sceneMin.z) / (sceneMax.z - sceneMin.z);

    // scale to [0, 1023] and expand to 30 bits
    double x = mMin(mMax(normalized.x * 1024.0, 0.0), 1023.0);
    double y = mMin(mMax(normalized.y * 1024.0, 0.0), 1023.0);
    double z = mMin(mMax(normalized.z * 1024.0, 0.0), 1023.0);
    unsigned int xx = expandBits((unsigned int)x);
    unsigned int yy = expandBits((unsigned int)y);
    unsigned int zz = expandBits((unsigned int)z);
    // printf("Morton code for (%f, %f, %f) = %u\n", normalized.x, normalized.y, normalized.z, xx * 4 + yy * 2 + zz);
    return xx * 4 + yy * 2 + zz;
}

struct Node
{
    int objectID = 0;
    bool isLeaf = false;
    AABB aabb;
    Node* childA = NULL;
    Node* childB = NULL;
};

// checks if the ray intersects the AABB
__device__ inline bool checkOverlap(const Ray& ray, const AABB& aabb, double t_min, double t_max)
{
    const double3 invD = make_double3(
        1.0 / ray.direction.x,
        1.0 / ray.direction.y,
        1.0 / ray.direction.z
    );

    const double3 orig = ray.origin;

    for (int i = 0; i < 3; i++)
    {
        double minVal = ((double*)&aabb.min)[i];
        double maxVal = ((double*)&aabb.max)[i];
        double o = ((double*)&orig)[i];
        double inv = ((double*)&invD)[i];

        double t0 = (fmin(minVal, maxVal) - o) * inv;
        double t1 = (fmax(minVal, maxVal) - o) * inv;

        if (inv < 0.0)
            mSwap(t0, t1);

        t_min = t0 > t_min ? t0 : t_min;
        t_max = t1 < t_max ? t1 : t_max;

        if (t_max <= t_min)
            return false;
    }

    return true;
}

__device__ inline int findSplit(unsigned int* sortedMortonCodes, int first, int last)
{
    // Identical Morton codes => split the range in the middle.

    unsigned int firstCode = sortedMortonCodes[first];
    unsigned int lastCode = sortedMortonCodes[last];

    if (firstCode == lastCode)
        return (first + last) >> 1;

    // Calculate the number of highest bits that are the same
    // for all objects, using the count-leading-zeros intrinsic.

    int commonPrefix = __clz(firstCode ^ lastCode);

    // Use binary search to find where the next bit differs.
    // Specifically, we are looking for the highest object that
    // shares more than commonPrefix bits with the first one.

    int split = first; // initial guess
    int step = last - first;

    do
    {
        step = (step + 1) >> 1; // exponential decrease
        int newSplit = split + step; // proposed new position

        if (newSplit < last)
        {
            unsigned int splitCode = sortedMortonCodes[newSplit];
            int splitPrefix = __clz(firstCode ^ splitCode);
            if (splitPrefix > commonPrefix)
                split = newSplit; // accept proposal
        }
    } while (step > 1);

    return split;
}

__device__ inline int delta(const unsigned int* mortonCodes, int numObjects, int i, int j)
{
    if (j < 0 || j >= numObjects)
        return -1; // 越界定义为最小公共前缀
    unsigned int codeA = mortonCodes[i];
    unsigned int codeB = mortonCodes[j];
    if (codeA == codeB) return 32; // 所有位都一样
    return __clz(codeA ^ codeB);   // 返回不同前的公共前缀长度
}

__device__ inline int2 determineRange(const unsigned int* mortonCodes, int numObjects, int idx)
{
    int d = 0;
    int deltaNext = delta(mortonCodes, numObjects, idx, idx + 1);
    int deltaPrev = delta(mortonCodes, numObjects, idx, idx - 1);

    // 第 3 行：决定搜索方向
    d = (deltaNext - deltaPrev) > 0 ? 1 : -1;

    // 第 5 行：δmin 是最小公共前缀
    int deltaMin = delta(mortonCodes, numObjects, idx, idx - d);

    // 第 6-8 行：指数扩展找最大范围
    int lMax = 2;
    while (delta(mortonCodes, numObjects, idx, idx + lMax * d) > deltaMin)
    {
        lMax *= 2;
    }

    // 第 9-14 行：二分查找精确范围端点 j
    int l = 0;
    for (int t = lMax / 2; t >= 1; t /= 2)
    {
        if (delta(mortonCodes, numObjects, idx, idx + (l + t) * d) > deltaMin)
        {
            l += t;
        }
    }

    int j = idx + l * d;

    // 第 15-21 行：找到子节点的划分点 γ（split point）
    int deltaNode = delta(mortonCodes, numObjects, idx, j);
    int s = 0;
    int span = abs(j - idx);

    for (int t = (span + 1) / 2; t >= 1; t /= 2)
    {
        if (delta(mortonCodes, numObjects, idx, idx + (s + t) * d) > deltaNode)
        {
            s += t;
        }
    }

    int gamma = idx + s * d + mMin(d, 0);

    // 返回区间 [first, last]
    int first = mMin(idx, j);
    int last = mMax(idx, j);
    return int2{ first, last };
}

__device__ inline void generateHierarchy(Hittable* d_objs, Node* leafNodes, Node* internalNodes, unsigned int* sortedMortonCodes, unsigned int* sortedObjectIDs, int num)
{
    // Construct leaf nodes.
    // Note: This step can be avoided by storing
    // the tree in a slightly different way.
    for (int idx = 0; idx < num; idx++)
    {
        leafNodes[idx].isLeaf = true;
        leafNodes[idx].objectID = sortedObjectIDs[idx];
        leafNodes[idx].aabb = d_objs[sortedObjectIDs[idx]].aabb;
        leafNodes[idx].childA = NULL;
        leafNodes[idx].childB = NULL;
        //printf("Leaf Node %d: AABB = [%f, %f, %f] - [%f, %f, %f]\n",
        //    leafNodes[idx].objectID,
        //    leafNodes[idx].aabb.min.x, leafNodes[idx].aabb.min.y, leafNodes[idx].aabb.min.z,
        //    leafNodes[idx].aabb.max.x, leafNodes[idx].aabb.max.y, leafNodes[idx].aabb.max.z
        //);
    }

    for (int idx = 0; idx < num - 1; idx++)
    {
        internalNodes[idx].isLeaf = false;
        internalNodes[idx].objectID = idx;
    }

    // Construct internal nodes.

    for (int idx = 0; idx < num - 1; idx++)
    {
        // Find out which range of objects the node corresponds to.
        // (This is where the magic happens!)

        int2 range = determineRange(sortedMortonCodes, num, idx);
        int first = range.x;
        int last = range.y;

        // Determine where to split the range.

        int split = findSplit(sortedMortonCodes, first, last);

        // Select childA.

        Node* childA;
        if (split == first)
            childA = &leafNodes[split];
        else
            childA = &internalNodes[split];

        // Select childB.

        Node* childB;
        if (split + 1 == last)
            childB = &leafNodes[split + 1];
        else
            childB = &internalNodes[split + 1];

        // Record parent-child relationships.

        internalNodes[idx].childA = childA;
        internalNodes[idx].childB = childB;

        //printf("Node %d / %d: AABB = %d(%d) - %d(%d)\n",
        //    internalNodes[idx].objectID,
        //    num - 1,
        //    internalNodes[idx].childA->objectID,
        //    internalNodes[idx].childA->isLeaf ? 1 : 0,
        //    internalNodes[idx].childB->objectID,
        //    internalNodes[idx].childB->isLeaf ? 1 : 0
        //);
    }

    // construct AABB

    Node* stack[256];
    Node** stackPtr = stack;
    Node* cur = internalNodes;
    int p_max = 0;
    int p = 0;
    Node* pre = NULL;

    while (cur != NULL || stackPtr != stack)
    {
        while (cur != NULL)
        {
            *stackPtr++ = cur;
            p_max = mMax(p_max, ++p);
            cur = cur->childA;
        }
        cur = *--stackPtr;
        p--;
        if (cur->childB == NULL || pre == cur->childB)
        {
            if (!cur->isLeaf)
            {
                cur->aabb = AABB(cur->childA->aabb, cur->childB->aabb);
                //printf("Node %d: AABB = [%f, %f, %f] - [%f, %f, %f]\n",
                //    cur->objectID,
                //    cur->aabb.min.x, cur->aabb.min.y, cur->aabb.min.z,
                //    cur->aabb.max.x, cur->aabb.max.y, cur->aabb.max.z);
            }
            pre = cur;
            cur = NULL;
        }
        else
        {
            *stackPtr++ = cur;
            p_max = mMax(p_max, ++p);
            cur = cur->childB;
        }
        //printf("max = %d\n", p_max);
    }
    //printf("max = %d\n", p_max);

}

__device__ inline int traverseIterative(Node* internalNodes, Hittable* objs, const Ray& ray, HitRecord& record)
{
    // Allocate traversal stack from thread-local memory,
    // and push NULL to indicate that there are no postponed nodes.
    Node* stack[64];
    Node** stackPtr = stack;
    *stackPtr++ = NULL; // push

    int hitId = -1;

    // Traverse nodes starting from the root.
    Node* node = internalNodes;
    double t_min = 0.001;
    double t_max = INF;
    HitRecord tempRecord;
    do
    {
        // Check each child node for overlap.
        Node* childL = node->childA;
        Node* childR = node->childB;
        bool overlapL = checkOverlap(ray, childL->aabb, t_min, t_max);
        bool overlapR = checkOverlap(ray, childR->aabb, t_min, t_max);

        // Query overlaps a leaf node => report collision.
        if (overlapL && childL->isLeaf)
            if (objs[childL->objectID].hit(ray, tempRecord, 0.001, hitId >= 0 ? record.t : INF))
            { 
                hitId = childL->objectID;
                record = tempRecord;
            }
        if (overlapR && childR->isLeaf)
            if (objs[childR->objectID].hit(ray, tempRecord, 0.001, hitId >= 0 ? record.t : INF))
            {
                hitId = childR->objectID;
                record = tempRecord;
            }

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
    
    return hitId;
}
