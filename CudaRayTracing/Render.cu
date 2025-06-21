
#include "Render.cuh"

#define CLAMP01(x) ((x) > 1.0 ? 1.0 : ((x) < 0.0 ? 0.0 : (x)))

__global__ void render(uchar4* devPtr, const Camera* camera, unsigned int* lightsIndex, Hittable** objs, Node* internalNodes, int lightsCount,
    int max_x, int max_y, int sampleCount, double roughness, double metallic, double t)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;

    // int pixelIndex = j * max_x * 4 + i * 4;
    int offset = j * max_x + i;
    
    // random sample
    curandState state;
    curand_init(3317, offset, 0, &state);

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
        double3 throughput = make_double3(1.0, 1.0, 1.0);  // ÀÛ³Ë fr * cos¦È / pdf
        double3 radiance = make_double3(0.0, 0.0, 0.0);    // final result
        // int depth = 0;
        while (true) 
        {
            // if no hit, break
            if (!traverseIterative(internalNodes, objs, ray, record))
            {
                radiance += throughput * camera->background;
                break;
            }
            if (record.material == NULL)
            {
                radiance += throughput * make_double3(10.0, 10.0, 10.0);
                break;
            }

            // contribution from the light source
            double3 direction;
            for (int k = 0; k < lightsCount; k++)
            {
                // if the material is glass, skip the light contribution
                if (record.material->glass)
                    break;

                Light light = (*objs)[lightsIndex[k]].light;
                double lightWidth = light.width;
                double lightHeight = light.width;
                double pdfLight = 1.0 / (lightWidth * lightHeight);
                double3 lightIntensity = light.color;
                double3 lightCenter = light.center;
                double3 lightPos = lightCenter +
                    (curand_uniform_double(&state) - 0.5) * lightWidth * light.edgeU +
                    (curand_uniform_double(&state) - 0.5) * lightHeight * light.edgeV;
                direction = Unit(lightPos - record.hitPos);
                Ray sampleRay = Ray(record.hitPos, direction, 0.0);
                if (Dot(direction, record.normal) > 0.0 && !traverseIterative(internalNodes, objs, sampleRay, tmp))
                {
                    record.material->roughness = roughness;
                    record.material->metallic = metallic;
                    double3 colorLight = lightIntensity * record.getFr(ray, direction) * Dot(direction, record.normal) * Dot(-direction, light.normal) / SquaredLength(lightPos - record.hitPos) / pdfLight;
                    radiance += throughput * colorLight;
                }
            }

            // contribution from other refectors
            // russian roulette
            double P_RR = 0.8;
            if (curand_uniform_double(&state) > P_RR)
                break;

            // depth
            // if (depth++ > 2) break;

            // randomly choose ONE direction w_i
            double r1 = curand_uniform_double(&state);
            double r2 = curand_uniform_double(&state);
            double pdf;
            record.getSample(ray, direction, pdf, r1, r2);
            record.material->roughness = roughness;
            record.material->metallic = metallic;
            throughput *= record.getFr(ray, direction) * Dot(direction, record.normal) / pdf / P_RR;
            // throughput *= fr * Dot(direction, record.normal) / pdf;

            ray = Ray(record.hitPos, direction, 0.0);
        }
        pixelRadience += radiance / sampleCount;
    }

    // set color
    unsigned char r = static_cast<unsigned char>(254.99 * CLAMP01(pixelRadience.x));
    unsigned char g = static_cast<unsigned char>(254.99 * CLAMP01(pixelRadience.y));
    unsigned char b = static_cast<unsigned char>(254.99 * CLAMP01(pixelRadience.z));
    devPtr[offset] = make_uchar4(r, g, b, 255);
}