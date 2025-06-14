
#include "Render.cuh"

#define CLAMP01(x) ((x) > 1.0 ? 1.0 : ((x) < 0.0 ? 0.0 : (x)))

__device__ bool isHitAnything(Hittable** objs, int obj_count, const Ray& ray, HitRecord& record)
{
    HitRecord tempRec;
    bool hitAnything = false;
    double closest = INF;

    for (int i = 0; i < obj_count; i++) 
    {
        if ((*objs)[i].hit(ray, tempRec, 0.001, closest)) 
        {
            hitAnything = true;
            closest = tempRec.t;
            record = tempRec;
        }
    }

    return hitAnything;
}


__global__ void render(unsigned char* cb, Camera* camera, Hittable** objs, int obj_count, int max_x, int max_y, double t) 
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;

    int pixel_index = j * max_x * 4 + i * 4;
    
    // random sample
    curandState state;
    curand_init(123, pixel_index, 0, &state);

    // path tracing
    int sampleCount = 500;

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
        while (true) 
        {
            // if no hit, break
            if (!isHitAnything(objs, obj_count, ray, record))
            {
                radiance += throughput * camera->background;
                break;
            }

            // contribution from the light source
            double lightWidth = 1.0;
            double lightHeight = 1.0;
            double pdfLight = 1.0 / (lightWidth * lightHeight);
            double3 lightCenter = make_double3(cos(t), 4.0, sin(t));
            double3 lightNormal = make_double3(0.0, -1.0, 0.0);
            // sample from the light source
            double3 lightPos = lightCenter + make_double3(
                (curand_uniform_double(&state) - 0.5) * lightWidth,
                0.0,
                (curand_uniform_double(&state) - 0.5) * lightHeight
            );
            double3 direction = Unit(lightPos - record.hitPos);
            ray = Ray(record.hitPos, direction, 0.0);
            if (Dot(direction, record.normal) > 0.0 && !isHitAnything(objs, obj_count, ray, tmp))
            {
                double3 fr = make_double3(10.0, 10.0, 10.0) / PI;
                double3 colorLight = fr * Dot(direction, record.normal) * Dot(-direction, lightNormal) / SquaredLength(lightPos - record.hitPos) / pdfLight;
                radiance += throughput * colorLight;
            }
            // contribution from other refectors
            // russian roulette
            double P_RR = 0.8;
            if (curand_uniform_double(&state) > P_RR) break;

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
            double3 fr = make_double3(0.8, 0.8, 0.8) / PI;
            double pdf = cosTheta / PI;
            throughput *= fr * Dot(direction, record.normal) / pdf / P_RR;

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