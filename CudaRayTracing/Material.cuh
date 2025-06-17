#pragma once

#include "Ray.cuh"

class Material
{
public:
    double3 color;
    double roughness;
    double metallic;

public:
    __host__ __device__ Material(const double3& _color)
        : color(_color), roughness(0.5), metallic(0.0)
    {
    }
    __host__ __device__ double3 fr(const Ray& ray, const double3& normal, const double3& direction)
    {
        double3 V = Unit(-ray.direction);   // 视线方向
        double3 L = Unit(direction);        // 光源方向
        double3 H = Unit(V + L);
        double3 F0;
        F0.x = (1 - metallic) * 0.04 + metallic * color.x;
        F0.y = (1 - metallic) * 0.04 + metallic * color.y;
        F0.z = (1 - metallic) * 0.04 + metallic * color.z;

        double D = DistributionGGX(normal, H, roughness);
        double G = GeometrySmith(normal, V, L, roughness);
        double3 F = FresnelSchlick(mMax(Dot(H, V), 0.0), F0);

        double NdotL = mMax(Dot(normal, L), 0.0);
        double NdotV = mMax(Dot(normal, V), 0.0);
        double denominator = 4.0 * NdotV * NdotL + 0.001;

        double3 specular = (D * G * F) / denominator;

        double3 kS = F;
        double3 kD = make_double3(1.0, 1.0, 1.0) - kS;
        kD *= 1.0 - metallic;

        double3 diffuse = kD * color / PI;

        return (diffuse + specular) * NdotL;
    }
    //double3 emitted() const
    //{
    //    return Color(0, 0, 0);
    //}

private:
    __host__ __device__ double DistributionGGX(double3 N, double3 H, double roughness)
    {
        double a = roughness * roughness;
        double a2 = a * a;
        double NdotH = mMax(Dot(N, H), 0.0);
        double NdotH2 = NdotH * NdotH;

        double denom = (NdotH2 * (a2 - 1.0) + 1.0);
        denom = PI * denom * denom;

        return a2 / denom;
    }
    __host__ __device__ double GeometrySchlickGGX(double NdotV, double roughness)
    {
        double r = roughness + 1.0;
        double k = (r * r) / 8.0;

        return NdotV / (NdotV * (1.0 - k) + k);
    }
    __host__ __device__ double GeometrySmith(double3 N, double3 V, double3 L, double roughness)
    {
        double NdotV = mMax(Dot(N, V), 0.0);
        double NdotL = mMax(Dot(N, L), 0.0);
        double ggx1 = GeometrySchlickGGX(NdotV, roughness);
        double ggx2 = GeometrySchlickGGX(NdotL, roughness);
        return ggx1 * ggx2;
    }
    __host__ __device__ double3 FresnelSchlick(double cosTheta, double3 F0)
    {
        return F0 + (make_double3(1.0, 1.0, 1.0) - F0) * pow(1.0 - cosTheta, 5.0);
    }
};