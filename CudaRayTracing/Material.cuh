#pragma once

#include "Ray.cuh"

class Material
{
public:
    double3 color;
    double roughness;
    double metallic;
    bool glass;  // 是否为玻璃材质

public:
    __host__ __device__ Material()
        : color(make_double3(0.0, 0.0, 0.0)), roughness(0.5), metallic(0.0), glass(false)
    {
    }
    __host__ __device__ Material(const double3 _color, bool _glass = false)
        : color(_color), roughness(0.5), metallic(0.0), glass(_glass)
    {
    }
    __host__ __device__ double3 fr(const Ray& ray, const double3 normal, const double3 direction)
    {
        if (glass) 
        {
            return make_double3(1.0, 1.0, 1.0);
        }

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

    __host__ __device__ void sampleLight(const Ray& ray, const double3 normal, double3& direction, double& pdf, double r1, double r2, bool frontFace)
    {
        if (glass)
        {
            double refIdx = 1.5; // 折射率
            double refRatio = frontFace ? (1.0 / refIdx) : refIdx;
            double3 unitRayDirection = Unit(ray.direction);
            double cosTheta = mMin(Dot(-unitRayDirection, normal), 1.0);
            // 全反射
            if (refRatio * sqrt(1 - cosTheta * cosTheta) > 1.0 ||
                Reflectance(cosTheta, refIdx) > r1)
            {
                direction = unitRayDirection - 2.0 * Dot(unitRayDirection, normal) * normal;
            }
            // 折射
            else
            {
                double3 vPerp = refRatio * (unitRayDirection + Dot(-unitRayDirection, normal) * normal);
                double3 vParallel = -sqrt(fabs(1.0 - SquaredLength(vPerp))) * normal;
                direction = vPerp + vParallel;
            }
            pdf = 1.0;
            return;
        }

        // cosine hemisphere sampling
        double sinTheta = sqrt(1.0 - r1);
        double cosTheta = sqrt(r1);
        // direction on ONB
        double3 w = normal;
        double3 a = (fabs(w.x) > 0.9) ? make_double3(0.0, 1.0, 0.0) : make_double3(1.0, 0.0, 0.0);
        double3 u = Unit(Cross(w, a));
        double3 v = Cross(w, u);
        direction = cos(2.0 * PI * r2) * sinTheta * u +
            sin(2.0 * PI * r2) * sinTheta * v +
            cosTheta * w;
        pdf = cosTheta / PI;
    }

    //double3 emitted() const
    //{
    //    return Color(0, 0, 0);
    //}

private:
    __host__ __device__ static double Reflectance(const double& cos, const double& refIdx)
    {
        double r0 = (1 - refIdx) / (1 + refIdx);
        r0 *= r0;
        return r0 + (1 - r0) * pow((1 - cos), 5);
    }

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