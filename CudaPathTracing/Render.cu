
#include "Render.cuh"

#define CLAMP01(x) ((x) > 1.0 ? 1.0 : ((x) < 0.0 ? 0.0 : (x)))

__global__ void clear(uchar4* devPtr, int max_x, int max_y)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;

    int offset = j * max_x + i;
    devPtr[offset] = make_uchar4(0, 0, 0, 255);
}

__global__ void render(uchar4* devPtr, uchar4* gBuffer, const Camera* camera, unsigned int* lightsIndex, Hittable* objs, Node* internalNodes, int lightsCount,
    int max_x, int max_y, int sampleCount, double t)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;

    // int pixelIndex = j * max_x * 4 + i * 4;
    int offset = j * max_x + i;
    
    // random sample
    curandState state;
    curand_init(1000, offset, 0, &state);

    // path tracing
   
    // pixel color
    double3 pixelRadience = make_double3(0.0, 0.0, 0.0);

    for (int sample = 0; sample < sampleCount; sample++)
    {
        // sample a random ray from the camera
        double randx = curand_uniform_double(&state);
        double randy = curand_uniform_double(&state);
        Ray ray = camera->getRandomSampleRay(i, j, randx, randy);
        HitRecord record;
        HitRecord tmp;

        // trace the ray
        double3 throughput = make_double3(1.0, 1.0, 1.0);  // 累乘 fr * cosθ / pdf
        double3 radiance = make_double3(0.0, 0.0, 0.0);    // final result
        bool firstHit = true;
        int depth = 0;
        while (true) 
        {
            // if no hit, break
            if (traverseIterative(internalNodes, objs, ray, record) < 0)
            {
                radiance += throughput * camera->background;
                break;
            }
            // if hit light
            if (record.material->type == MaterialType::M_LIGHT)
            {
                radiance += throughput * record.material->color;
                break;
            }

            // contribution from the light source
            double3 direction;
            for (int k = 0; k < lightsCount; k++)
            {
                // if the material is glass, skip the light contribution
                if (record.material->type != MaterialType::M_OPAQUE)
                    break;

                Light light = objs[lightsIndex[k]].light;
                double lightWidth = light.width;
                double lightHeight = light.width;
                double pdfLight = 1.0 / (lightWidth * lightHeight);
                double3 lightIntensity = light.material.color;
                double3 lightCenter = light.center;
                double3 lightPos = lightCenter +
                    (curand_uniform_double(&state) - 0.5) * lightWidth * light.edgeU +
                    (curand_uniform_double(&state) - 0.5) * lightHeight * light.edgeV;
                direction = Unit(lightPos - record.hitPos);
                Ray sampleRay = Ray(record.hitPos, direction, 0.0);

                int obstacleToLight = traverseIterative(internalNodes, objs, sampleRay, tmp);
                if (Dot(direction, record.normal) > 0.0 && Dot(-direction, light.normal) > 0.0 && obstacleToLight == lightsIndex[k])
                {
                    double3 colorLight = lightIntensity * record.getFr(ray, direction) * Dot(direction, record.normal) * Dot(-direction, light.normal) / SquaredLength(lightPos - record.hitPos) / pdfLight;
                    radiance += throughput * colorLight;
                }
            }

            // contribution from other refectors
            // russian roulette
            double P_RR = 0.7;
            if (curand_uniform_double(&state) > P_RR)
                break;

            // depth
            //double P_RR = 1.0;
            //if (depth++ > 0) break;

            // randomly choose ONE direction w_i
            double r1 = curand_uniform_double(&state);
            double r2 = curand_uniform_double(&state);
            double r3 = curand_uniform_double(&state);
            double pdf;
            double3 wm;
            record.getSample(ray, direction, wm, pdf, r1, r2, r3);
            if (pdf <= 0.0)
                break;
            // ! IMPORTANT, cosθ_i在折射为负数，需要加上绝对值，否则折射全黑
            throughput *= record.getFr(ray, direction, wm) * fabs(Dot(direction, record.normal)) / pdf / P_RR;

            ray = Ray(record.hitPos, direction, 0.0);
            firstHit = false;
        }
        pixelRadience += radiance;
    }
    pixelRadience /= sampleCount;

    // gamma
    pixelRadience.x = sqrt(pixelRadience.x);
    pixelRadience.y = sqrt(pixelRadience.y);
    pixelRadience.z = sqrt(pixelRadience.z);
    // set color
    unsigned char r = static_cast<unsigned char>(254.99 * CLAMP01(pixelRadience.x));
    unsigned char g = static_cast<unsigned char>(254.99 * CLAMP01(pixelRadience.y));
    unsigned char b = static_cast<unsigned char>(254.99 * CLAMP01(pixelRadience.z));

    gBuffer[offset] = devPtr[offset];
    devPtr[offset] = make_uchar4(r, g, b, 255);
}

__global__ void gaussian(uchar4* devPtr, int max_x, int max_y)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i <= 0 || i >= max_x - 1 || j <= 0 || j >= max_y - 1) return;

    uchar4 result = make_uchar4(0, 0, 0, 255);
    double gaussianKernel[3][3] = {
        {1.0 / 16, 2.0 / 16, 1.0 / 16},
        {2.0 / 16, 4.0 / 16, 2.0 / 16},
        {1.0 / 16, 2.0 / 16, 1.0 / 16}
    };

    // 3x3 高斯滤波
    for (int dy = -1; dy <= 1; dy++) 
    {
        for (int dx = -1; dx <= 1; dx++) 
        {
            int ix = i + dx;
            int iy = j + dy;
            int offset = iy * max_x + ix;
            result.x += devPtr[offset].x * gaussianKernel[dy + 1][dx + 1];
            result.y += devPtr[offset].y * gaussianKernel[dy + 1][dx + 1];
            result.z += devPtr[offset].z * gaussianKernel[dy + 1][dx + 1];
        }
    }

    int offset = j * max_x + i;
    devPtr[offset] = result;
}

__global__ void addPrevious(uchar4* devPtr, uchar4* gBuffer, int max_x, int max_y)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;

    int offset = j * max_x + i;

    double alpha = 0.2;
    devPtr[offset].x = devPtr[offset].x * 0.2 + gBuffer[offset].x * (1.0 - alpha);
    devPtr[offset].y = devPtr[offset].y * 0.2 + gBuffer[offset].y * (1.0 - alpha);
    devPtr[offset].z = devPtr[offset].z * 0.2 + gBuffer[offset].z * (1.0 - alpha);
}



__global__ void getObject(Hittable* objs, const Camera* camera, Node* internalNodes, int* selectPtr, const int x, const int y)
{
    HitRecord record;
    Ray ray = camera->getSampleRay(x, y);
    int hitId = traverseIterative(internalNodes, objs, ray, record);
    if (hitId >= 0 && record.material != NULL)
    {
        *selectPtr = hitId;
    }
    else
    {
        *selectPtr = -1;
    }
}

__global__ void changeMaterial(Hittable* objs, const int start, const int end, const double alphaX, const double alphaY, const bool glass)
{
    int idx = start + blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < start || idx >= end) return;

    switch (objs[idx].type)
    {
    case ObjectType::SPHERE:
        objs[idx].sphere.material.alphaX = alphaX;
        objs[idx].sphere.material.alphaY = alphaY;
        objs[idx].sphere.material.type = glass ? MaterialType::M_SPECULAR_DIELECTRIC : MaterialType::M_OPAQUE;
        break;
    case ObjectType::TRIANGLE:
        objs[idx].triangle.material.alphaX = alphaX;
        objs[idx].triangle.material.alphaY = alphaY;
        objs[idx].triangle.material.type = glass ? MaterialType::M_SPECULAR_DIELECTRIC : MaterialType::M_OPAQUE;
        break;
    case ObjectType::LIGHT:
    case ObjectType::NONE:
    default:
        return;
    }
}