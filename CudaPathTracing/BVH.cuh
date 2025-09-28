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
__device__ inline uint64_t morton3D(double3 v, const double3 sceneMin, const double3 sceneMax, int index)
{
    // normalize
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
    uint64_t morton = xx * 4 + yy * 2 + zz;
    return (morton << 32) | index;
}

struct Node
{
    int objectID = 0;
    bool isLeaf = false;
    AABB aabb;
    Node* childA = NULL;
    Node* childB = NULL;
    int visited = 0;
};

// checks if the ray intersects the AABB
__device__ inline bool checkSlab(
    double o, double inv, double minVal, double maxVal,
    double& t_min, double& t_max)
{
    double t0 = (minVal - o) * inv;
    double t1 = (maxVal - o) * inv;
    if (inv < 0.0)
        mSwap(t0, t1);
    t_min = fmax(t0, t_min);
    t_max = fmin(t1, t_max);
    return t_max > t_min;
}
__device__ inline bool checkOverlap(const Ray& ray, const AABB& aabb, double t_min, double t_max)
{
    const double3 invD = make_double3(
        1.0 / ray.direction.x,
        1.0 / ray.direction.y,
        1.0 / ray.direction.z
    );

    const double3 orig = ray.origin;

    return checkSlab(orig.x, invD.x, aabb.min.x, aabb.max.x, t_min, t_max) &&
        checkSlab(orig.y, invD.y, aabb.min.y, aabb.max.y, t_min, t_max) &&
        checkSlab(orig.z, invD.z, aabb.min.z, aabb.max.z, t_min, t_max);
}

__device__ inline int findSplit(uint64_t* sortedMortonCodes, int first, int last)
{
    // Identical Morton codes => split the range in the middle.

    uint64_t firstCode = sortedMortonCodes[first];
    uint64_t lastCode = sortedMortonCodes[last];

    if (firstCode == lastCode)
        return (first + last) >> 1;

    // Calculate the number of highest bits that are the same
    // for all objects, using the count-leading-zeros intrinsic.

    int commonPrefix = __clzll(firstCode ^ lastCode);

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
            uint64_t splitCode = sortedMortonCodes[newSplit];
            int splitPrefix = __clzll(firstCode ^ splitCode);
            if (splitPrefix > commonPrefix)
                split = newSplit; // accept proposal
        }
    } while (step > 1);

    return split;
}

__device__ inline int delta(const uint64_t* mortonCodes, int numObjects, int i, int j)
{
    if (j < 0 || j >= numObjects)
        return -1; // 越界定义为最小公共前缀
    uint64_t codeA = mortonCodes[i];
    uint64_t codeB = mortonCodes[j];
    if (codeA == codeB) return 64; // 所有位都一样
    return __clzll(codeA ^ codeB);   // 返回不同前的公共前缀长度
}

__device__ inline int2 determineRange(const uint64_t* mortonCodes, int numObjects, int idx)
{
    int d = 0;
    int deltaNext = delta(mortonCodes, numObjects, idx, idx + 1);
    int deltaPrev = delta(mortonCodes, numObjects, idx, idx - 1);        

    // 决定搜索方向
    d = (deltaNext > deltaPrev) ? 1 : -1;

    // δmin 是最小公共前缀
    int deltaMin = delta(mortonCodes, numObjects, idx, idx - d);

    // 指数扩展找最大范围
    int lMax = 2;
    while (delta(mortonCodes, numObjects, idx, idx + lMax * d) > deltaMin)
    {
        lMax *= 2;
    }
    // 二分查找精确范围端点 j
    int l = 0;
    for (int t = lMax / 2; t >= 1; t /= 2)
    {
        if (delta(mortonCodes, numObjects, idx, idx + (l + t) * d) > deltaMin)
        {
            l += t;
        }
    }

    int j = idx + l * d;

    //if (l == 0)
    //{
    //    while (delta(mortonCodes, numObjects, idx, idx + l * d) == deltaMin)
    //        l++;
    //    l--;
    //    j = idx + l * d;
    //}

    // 返回区间 [first, last]
    int first = mMin(idx, j);
    int last = mMax(idx, j);
    return int2{ first, last };
}

__device__ inline void generateHierarchy(Hittable* d_objs, Node* leafNodes, Node* internalNodes, uint64_t* sortedMortonCodes, unsigned int* sortedObjectIDs, int num, int idx)
{
    // Construct leaf nodes.
    // Note: This step can be avoided by storing
    // the tree in a slightly different way.

    leafNodes[idx].isLeaf = true;
    leafNodes[idx].objectID = sortedObjectIDs[idx];
    leafNodes[idx].aabb = d_objs[sortedObjectIDs[idx]].aabb;
    leafNodes[idx].childA = NULL;
    leafNodes[idx].childB = NULL;
    //printf("Leaf Node %d: AABB = [%f, %f, %f] - [%f, %f, %f] : %d\n",
    //    leafNodes[idx].objectID,
    //    leafNodes[idx].aabb.min.x, leafNodes[idx].aabb.min.y, leafNodes[idx].aabb.min.z,
    //    leafNodes[idx].aabb.max.x, leafNodes[idx].aabb.max.y, leafNodes[idx].aabb.max.z,
    //    sortedMortonCodes[idx]
    //);

    if (idx >= num - 1)
        return;

    internalNodes[idx].isLeaf = false;
    internalNodes[idx].objectID = idx;
    internalNodes[idx].childA = NULL;
    internalNodes[idx].childB = NULL;
    internalNodes[idx].visited = 0;

    // Construct internal nodes.
    // 
    // Find out which range of objects the node corresponds to.
    // (This is where the magic happens!)
    int2 range = determineRange(sortedMortonCodes, num, idx);
    int first = range.x;
    int last = range.y;

    // Determine where to split the range.
    int split = findSplit(sortedMortonCodes, first, last);
    //printf("%d: [%d, %d], %d\n", idx, first, last, split);

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
    childA->visited = 1;
    childB->visited = 1;

    //printf("Node %d / %d: AABB = %d(%d) - %d(%d)\n",
    //    internalNodes[idx].objectID,
    //    num - 1,
    //    internalNodes[idx].childA->objectID,
    //    internalNodes[idx].childA->isLeaf ? 1 : 0,
    //    internalNodes[idx].childB->objectID,
    //    internalNodes[idx].childB->isLeaf ? 1 : 0
    //);
}

__global__ inline void constructAABB(Node* internalNodes)
{
    // construct AABB
    Node* stack[512];
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
            cur = cur->childA;
            p_max = mMax(p_max, ++p);
        }
        cur = *--stackPtr;
        p--;
        if (cur->childB == NULL || pre == cur->childB)
        {
            if (!cur->isLeaf)
            {
                cur->aabb = AABB(cur->childA->aabb, cur->childB->aabb);
            }
            pre = cur;
            cur = NULL;
        }
        else
        {
            *stackPtr++ = cur;
            cur = cur->childB;
            p_max = mMax(p_max, ++p);
        }
    }
    printf("max = %d\n", p_max);
}

__device__ inline int traverseIterative(Node* internalNodes, Hittable* objs, const Ray& ray, HitRecord& record, double tMax = INF, double tMin = 0.001)
{
    // Allocate traversal stack from thread-local memory,
    // and push NULL to indicate that there are no postponed nodes.
    Node* stack[512];
    Node** stackPtr = stack;
    *stackPtr++ = NULL; // push

    int hitId = -1;

    // Traverse nodes starting from the root.
    Node* node = internalNodes;
    HitRecord tempRecord;
    do
    {
        // Check each child node for overlap.
        Node* childL = node->childA;
        Node* childR = node->childB;
        bool overlapL = checkOverlap(ray, childL->aabb, tMin, tMax);
        bool overlapR = checkOverlap(ray, childR->aabb, tMin, tMax);

        // Query overlaps a leaf node => report collision.
        if (overlapL && childL->isLeaf)
            if (objs[childL->objectID].hit(ray, tempRecord, 0.001, hitId >= 0 ? record.t : tMax))
            { 
                hitId = childL->objectID;
                record = tempRecord;
            }
        if (overlapR && childR->isLeaf)
            if (objs[childR->objectID].hit(ray, tempRecord, 0.001, hitId >= 0 ? record.t : tMax))
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