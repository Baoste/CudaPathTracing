#pragma once

#include "Ray.cuh"

class Material
{
public:
    double3 color;
    double alphaX;
    double alphaY;
    bool glass;  // 是否为玻璃材质
    double eta = 1.51;

public:
    __host__ __device__ Material()
        : color(make_double3(0.0, 0.0, 0.0)), alphaX(0.5), alphaY(0.5), glass(false)
    {
    }
    __host__ __device__ Material(const double3 _color, double _alphaX, double _alphaY, bool _glass = false)
        : color(_color), alphaX(_alphaX), alphaY(_alphaY), glass(_glass)
    {
    }
    __host__ __device__ double3 fr(const Ray& ray, const double3 normal, const double3 direction, bool frontFace)
    {
        double3 V = Unit(-ray.direction);   // 视线方向
        double3 L = Unit(direction);        // 光源方向
        double3 H = Unit(V + L);
        
        if (glass)
        {
            // BTDF
            //double refRatio = frontFace ? (1.0 / eta) : eta;
            //double F = FresnelDielectric(normal, V, L, refRatio);
            //double cosTheta_i = fabs(Dot(normal, V));
            //double cosTheta_t = fabs(Dot(normal, L));
            //double denominator = refRatio * cosTheta_i + cosTheta_t;
            //double specular = ((1.0 - F) * refRatio * refRatio) / (denominator * denominator);
            //return  specular * make_double3(1.0, 1.0, 1.0);
            double cosTheta_t = fabs(Dot(normal, L));
            return  color / cosTheta_t;
        }
        else
        {
            // BRDF
            double n = 2.63;
            double k = 9.08;

            double D = DistributionGGX(normal, H);
            double G = GeometrySmith(normal, V, L);
            double F = FresnelConductor(normal, V, L, n, k);

            double NdotL = mMax(Dot(normal, L), 0.0);
            double NdotV = mMax(Dot(normal, V), 0.0);
            double denominator = 4.0 * NdotV * NdotL + 0.001;

            double specular = (D * G * F) / denominator;
            return specular * color;
        }
    }

    __host__ __device__ void sampleLight(const Ray& ray, const double3 normal, double3& direction, double& pdf, double r1, double r2, bool frontFace)
    {
        if (glass)
        {
            double refRatio = frontFace ? (1.0 / eta) : eta;
            double3 unitRayDirection = Unit(ray.direction);
            double cosTheta = mMin(Dot(-unitRayDirection, normal), 1.0);
            // 全反射
            if (refRatio * sqrt(1 - cosTheta * cosTheta) > 1.0)
            {
                direction = unitRayDirection - 2.0 * Dot(unitRayDirection, normal) * normal;
                pdf = 1.0;
                return;
            }

            double3 vPerp = refRatio * (unitRayDirection + Dot(-unitRayDirection, normal) * normal);
            double3 vParallel = -sqrt(fabs(1.0 - SquaredLength(vPerp))) * normal;
            double3 refractDir = vPerp + vParallel;
            double R = FresnelDielectric(normal, -unitRayDirection, Unit(refractDir), refRatio);
            // 菲涅尔反射
            if (r1 < R)
            {
                direction = unitRayDirection - 2.0 * Dot(unitRayDirection, normal) * normal;
                pdf = 1.0;
            }
            // 折射
            else
            {
                direction = refractDir;
                pdf = 1.0;
            }
            return;
        }

        // cosine hemisphere sampling
        //double sinTheta = sqrt(1.0 - r1);
        //double cosTheta = sqrt(r1);
        //// direction on ONB
        //double3 w = normal;
        //double3 a = (fabs(w.x) > 0.9) ? make_double3(0.0, 1.0, 0.0) : make_double3(1.0, 0.0, 0.0);
        //double3 u = Unit(Cross(w, a));
        //double3 v = Cross(w, u);
        //direction = cos(2.0 * PI * r2) * sinTheta * u +
        //    sin(2.0 * PI * r2) * sinTheta * v +
        //    cosTheta * w;
        //pdf = cosTheta / PI;
        
        // BRDF sampling
        double3 w = normal;
        double3 a = (fabs(w.x) > 0.9) ? make_double3(0.0, 1.0, 0.0) : make_double3(1.0, 0.0, 0.0);
        double3 u = Unit(Cross(w, a));
        double3 v = Cross(w, u);

        double3 unitRayDirection = -Unit(ray.direction);
        double3 W_o = make_double3(Dot(unitRayDirection, u), Dot(unitRayDirection, v), Dot(unitRayDirection, w));
        double3 W_h = Unit(make_double3(alphaX * W_o.x, alphaY * W_o.y, W_o.z));
        double3 T1 = (W_h.z < 0.99999f) ? Unit(Cross(make_double3(0, 0, 1), W_h)) : make_double3(1, 0, 0);
        double3 T2 = Cross(W_h, T1);
        double r = sqrt(r1);
        double theta = 2 * PI * r2;
        double2 p = make_double2(r * cos(theta), r * sin(theta));
        double h = sqrt(1 - p.x * p.x);
        // Lerp((1 + wh.z) / 2, h, p.y);
        double lerpX = (1 + W_h.z) / 2.0;
        p.y = (1.0 - lerpX) * h + lerpX * p.y;
        double pz = sqrt(mMax(0.0, 1.0 - (p.x * p.x + p.y * p.y)));
        double3 nh = p.x * T1 + p.y * T2 + pz * W_h;
        double3 W_m = Unit(make_double3(alphaX * nh.x, alphaY * nh.y, mMax(1e-6f, nh.z)));
        double3 W_i = -W_o + 2 * Dot(W_o, W_m) * W_m;
        direction = Unit(W_i.x * u + W_i.y * v + W_i.z * w);
        double3 H = Unit((direction + unitRayDirection));
        pdf = GeometrySmith(normal, unitRayDirection) / W_o.z * DistributionGGX(normal, H) * Dot(W_o, W_m) / (4 * Dot(W_o, W_m));

    }

    //double3 emitted() const
    //{
    //    return Color(0, 0, 0);
    //}

private:
    __host__ __device__ static double reflectance(const double& cos, const double& refIdx)
    {
        double r0 = (1 - refIdx) / (1 + refIdx);
        r0 *= r0;
        return r0 + (1 - r0) * pow((1 - cos), 5);
    }

    __host__ __device__ double3 worldToLocal(double3 N, double3 W)
    {
        double3 zAxis = N;
        double3 up = make_double3(0.0, 1.0, 0.0);
        if (fabs(Dot(zAxis, up)) > 0.999f)
            up = make_double3(1.0, 0.0, 0.0);
        double3 xAxis = Unit(Cross(up, zAxis));
        double3 yAxis = Cross(zAxis, xAxis);

        return make_double3(Dot(W, xAxis), Dot(W, yAxis), Dot(W, zAxis));
    }

    __host__ __device__ double DistributionGGX(double3 N, double3 H)
    {
        double3 w = worldToLocal(N, H);
        double sinTheta = sqrt(1 - w.z * w.z);
        double cosPhi = w.x / sinTheta;
        double sinPhi = w.y / sinTheta;

        double theta = acos(mMax(Dot(N, H), 0.0));
        double tan2Theta = pow(tan(theta), 2);
        double cos4Theta = pow(cos(theta), 4);
        double e = tan2Theta * (pow(cosPhi / alphaX, 2) + pow(sinPhi / alphaY, 2));
        return 1.0 / (PI * alphaX * alphaY * cos4Theta * (1 + e) * (1 + e));
    }
    __host__ __device__ double lambda(double3 N, double3 R)
    {
        double3 w = worldToLocal(N, R);
        double sinTheta = sqrt(1 - w.z * w.z);
        double cosPhi = w.x / sinTheta;
        double sinphi = w.y / sinTheta;
        double alpha2 = alphaX * alphaX * cosPhi * cosPhi + alphaY * alphaY * sinphi * sinphi;
        double alpha = sqrt(alpha2);

        double tanTheta = tan(acos(mMax(Dot(N, R), 0.0)));
        return (sqrt(1.0 + alpha * alpha * tanTheta * tanTheta) - 1.0) / 2.0;
    }
    __host__ __device__ double GeometrySmith(double3 N, double3 V, double3 L)
    {
        return 1.0 / (1.0 + lambda(N, V) + lambda(N, L));
    }
    __host__ __device__ double GeometrySmith(double3 N, double3 W)
    {
        return 1.0 / (1.0 + lambda(N, W));
    }
    // Conductors Fresnel
    __host__ __device__ double FresnelConductor(double3 N, double3 V, double3 L, double n, double k)
    {
        Complex eta = make_double2(n, k);
        double cosTheta_i = mMax(Dot(N, V), 0.0);
        double cosTheta_t = mMax(Dot(N, L), 0.0);
        Complex r_parl = (eta * cosTheta_i - cosTheta_t) / (eta * cosTheta_i + cosTheta_t);
        Complex r_perp = (cosTheta_i - eta * cosTheta_t) / (cosTheta_i + eta * cosTheta_t);
        return (Norm(r_parl) + Norm(r_perp)) / 2.0;
    }
    // Transmission Fresnel
    __host__ __device__ double FresnelDielectric(double3 N, double3 V, double3 L, double eta)
    {
        double cosTheta_i = fabs(Dot(N, V));
        double cosTheta_t = fabs(Dot(N, L));
        double r_parl = (eta * cosTheta_i - cosTheta_t) / (eta * cosTheta_i + cosTheta_t);
        double r_perp = (cosTheta_i - eta * cosTheta_t) / (cosTheta_i + eta * cosTheta_t);
        return (r_parl * r_parl + r_perp * r_perp) / 2.0;
    }
};