
#include "Render.cuh"

#define CLAMP01(x) ((x) > 1.0 ? 1.0 : ((x) < 0.0 ? 0.0 : (x)))

__global__ void render(unsigned char* cb, const Camera* camera, unsigned int* lightsIndex, Hittable** objs, Node* internalNodes, int lightsCount, int max_x, int max_y, int sampleCount, double t)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;

    int pixel_index = j * max_x * 4 + i * 4;
    
    // random sample
    curandState state;
    curand_init(3317, pixel_index, 0, &state);

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

            // contribution from the light source
            double3 direction;
            for (int k = 0; k < lightsCount; k++)
            {
                Light light = (*objs)[lightsIndex[k]].light;
                double lightWidth = light.width;
                double lightHeight = light.width;
                double pdfLight = 1.0 / (lightWidth * lightHeight);
                double3 lightIntensity = light.color;
                double3 lightCenter = light.center + make_double3(2.0 * cos(t), 0, 0);
                double3 lightNormal = light.normal;
                // sample from the light source
                double3 lightPos = lightCenter + make_double3(
                    (curand_uniform_double(&state) - 0.5) * lightWidth,
                    0.0,
                    (curand_uniform_double(&state) - 0.5) * lightHeight
                );
                direction = Unit(lightPos - record.hitPos);
                Ray sampleRay = Ray(record.hitPos, direction, 0.0);
                if (Dot(direction, record.normal) > 0.0 && !traverseIterative(internalNodes, objs, sampleRay, tmp))
                {
                    double3 colorLight = lightIntensity * record.getFr(ray, direction) * Dot(direction, record.normal) * Dot(-direction, lightNormal) / SquaredLength(lightPos - record.hitPos) / pdfLight;
                    radiance += throughput * colorLight;
                }
            }

            // contribution from other refectors
            // russian roulette
            double P_RR = 0.8;
            if (curand_uniform_double(&state) > P_RR) break;

            // depth
            // if (depth++ > 2) break;

            // randomly choose ONE direction w_i
            // cosine hemisphere sampling
            double r1 = curand_uniform_double(&state);
            double r2 = curand_uniform_double(&state);
            double sinTheta = sqrt(1.0 - r1);
            double cosTheta = sqrt(r1);
            // direction on ONB
            double3 w = record.normal;
            double3 a = (fabs(w.x) > 0.9) ? make_double3(0.0, 1.0, 0.0) : make_double3(1.0, 0.0, 0.0);
            double3 u = Unit(Cross(w, a));
            double3 v = Cross(w, u);
            direction = cos(2.0 * PI * r2) * sinTheta * u +
                sin(2.0 * PI * r2) * sinTheta * v +
                cosTheta * w;
            double pdf = cosTheta / PI;
            throughput *= record.getFr(ray, direction) * Dot(direction, record.normal) / pdf / P_RR;
            // throughput *= fr * Dot(direction, record.normal) / pdf;

            ray = Ray(record.hitPos, direction, 0.0);
        }
        pixelRadience += radiance / sampleCount;
    }

    // set color
    cb[pixel_index + 0] = static_cast<unsigned char>(255.99 * CLAMP01(pixelRadience.x));
    cb[pixel_index + 1] = static_cast<unsigned char>(255.99 * CLAMP01(pixelRadience.y));
    cb[pixel_index + 2] = static_cast<unsigned char>(255.99 * CLAMP01(pixelRadience.z));
    cb[pixel_index + 3] = 255;
}