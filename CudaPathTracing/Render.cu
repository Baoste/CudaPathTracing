
#include "Render.cuh"

#define CLAMP01(x) ((x) > 1.0 ? 1.0 : ((x) < 0.0 ? 0.0 : (x)))

__global__ void clear(double3* pic, int max_x, int max_y)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;

    int offset = j * max_x + i;
    pic[offset] = make_double3(0.0, 0.0, 0.0);
}

__global__ void render(uchar4* devPtr, double3* pic, double3* picPrevious, double3* picBeforeGussian, double4* gBuffer, double3* gBufferPosition,
    const Camera* camera, unsigned int* lightsIndex, Hittable* objs, Node* internalNodes, int lightsCount,
    int max_x, int max_y, int sampleCount, double t, RenderType rederType)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;

    // int pixelIndex = j * max_x * 4 + i * 4;
    int offset = j * max_x + i;
    
    // random sample
    curandState state;
    double seed = 3317.0;
    if (rederType == RenderType::REAL_TIME)
        seed = 1000.0 * t;
    curand_init(seed, offset, 0, &state);

    // path tracing
   
    // pixel color
    double3 pixelRadience = make_double3(0.0, 0.0, 0.0);
    if (rederType == RenderType::NORMAL || rederType == RenderType::DEPTH) sampleCount = 1;

    for (int sample = 0; sample < sampleCount; sample++)
    {
        // sample a random ray from the camera
        double randx = curand_uniform_double(&state);
        double randy = curand_uniform_double(&state);
        Ray ray = camera->getRandomSampleRay(i, j, randx, randy);
        // trace the ray
        double3 throughput = make_double3(1.0, 1.0, 1.0);  // 累乘 fr * cosθ / pdf
        double3 radiance = make_double3(0.0, 0.0, 0.0);    // final result
        int depth = 0;
        
        // init G-buffer
        gBuffer[offset] = make_double4(0.0, 0.0, 0.0, INF);
        gBufferPosition[offset] = make_double3(INF, INF, INF);

        while (true) 
        {
            HitRecord record;
            HitRecord lightRecord;

            // if no hit, break
            if (traverseIterative(internalNodes, objs, ray, record) < 0)
            {
                radiance += throughput * camera->background;
                break;
            }
            // if hit light
            if (record.material->type == MaterialType::M_LIGHT)
            {
                radiance += throughput * record.hitColor;
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
                double3 lightCenter = light.center;
                double3 lightPos = lightCenter +
                    (curand_uniform_double(&state) - 0.5) * lightWidth * light.edgeU +
                    (curand_uniform_double(&state) - 0.5) * lightHeight * light.edgeV;
                direction = Unit(lightPos - record.hitPos);
                Ray sampleRay = Ray(record.hitPos, direction, 0.0);

                int obstacleToLight = traverseIterative(internalNodes, objs, sampleRay, lightRecord);
                if (Dot(direction, record.normal) > 0.0 && Dot(-direction, light.normal) > 0.0 && obstacleToLight == lightsIndex[k])
                {
                    double3 colorLight = lightRecord.hitColor * record.getFr(ray, direction) * Dot(direction, record.normal) * Dot(-direction, light.normal) / SquaredLength(lightPos - record.hitPos) / pdfLight;
                    radiance += throughput * colorLight;
                }
            }

            // contribution from other refectors
            double P_RR = 1.0;
            if (rederType == RenderType::REAL_TIME)
            {
                if (depth++ >= 2) break;
            }
            else if (rederType == RenderType::NORMAL || rederType == RenderType::DEPTH)
            {
                if (depth++ >= 1) break;
            }
            else
            {
                // russian roulette
                P_RR = 0.7;
                if (curand_uniform_double(&state) > P_RR) break;
            }

            // G-buffer
            if (depth == 1)
            {
                gBuffer[offset].x = record.normal.x;
                gBuffer[offset].y = record.normal.y;
                gBuffer[offset].z = record.normal.z;
                gBuffer[offset].w = Length(camera->lookFrom - record.hitPos);
                gBufferPosition[offset] = record.hitPos;
            }

            // randomly choose ONE direction w_i
            double r1 = curand_uniform_double(&state);
            double r2 = curand_uniform_double(&state);
            double r3 = curand_uniform_double(&state);
            double pdf;

            record.getSample(ray, direction, pdf, r1, r2, r3);
            if (pdf <= 0.0)
                break;
            // ! IMPORTANT, cosθ_i在折射为负数，需要加上绝对值，否则折射全黑
            throughput *= record.getFr(ray, direction) * fabs(Dot(direction, record.normal)) / pdf / P_RR;

            ray = Ray(record.hitPos, direction, 0.0);
        }
        pixelRadience += radiance;
    }
    pixelRadience /= sampleCount;

    picPrevious[offset] = pic[offset];
    pic[offset] = pixelRadience;
    picBeforeGussian[offset] = pixelRadience;

    // 实时观看渲染过程
    double3 pixel;
    switch (rederType)
    {
    case RenderType::STATIC:
        pixel = pic[offset];
        // gamma
        pixel.x = sqrt(pixel.x);
        pixel.y = sqrt(pixel.y);
        pixel.z = sqrt(pixel.z);
        break;
    case RenderType::NORMAL:
        pixel.x = gBuffer[offset].x;
        pixel.y = gBuffer[offset].y;
        pixel.z = gBuffer[offset].z;
        pixel = (pixel + make_double3(0.5, 0.5, 0.5)) / 2.0;
        break;
    case RenderType::DEPTH:
        double far = 50.0;
        double fragColor = gBuffer[offset].w / far;
        fragColor = CLAMP01(fragColor);
        pixel.x = fragColor;
        pixel.y = fragColor;
        pixel.z = fragColor;
        break;
    default:
        return;
    }
    unsigned char r = static_cast<unsigned char>(254.99 * CLAMP01(pixel.x));
    unsigned char g = static_cast<unsigned char>(254.99 * CLAMP01(pixel.y));
    unsigned char b = static_cast<unsigned char>(254.99 * CLAMP01(pixel.z));

    devPtr[offset] = make_uchar4(r, g, b, 255);
}

__global__ void copyPic(double3* target, double3* source, int max_x, int max_y)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= max_x || y >= max_y) return;

    int offset = y * max_x + x;
    target[offset] = source[offset];
}

//__global__ void gaussianSeparate(uchar4* devPtr, uchar4* devBeforeGussian, double* d_kernel, int max_x, int max_y)
//{
//    int x = blockIdx.x * blockDim.x + threadIdx.x;
//    int y = blockIdx.y * blockDim.y + threadIdx.y;
//    if (x >= max_x || y >= max_y) return;
//
//    int offset = y * max_x + x;
//
//    // 横向
//    double3 result = make_double3(0.0, 0.0, 0.0);
//    double sumWeight = 0.0;
//    for (int t = 1; t < 1 << KERNEL_SIZE; t <<= 1)
//    {
//        for (int i = -KERNEL_RADIUS; i <= KERNEL_RADIUS; i++)
//        {
//            for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
//            {
//                int nx = mMin(mMax(x + t * i, 0), max_x - 1);
//                int ny = mMin(mMax(y + t * j, 0), max_y - 1);
//
//                uchar4 pixelRGBW = devBeforeGussian[ny * max_x + nx];
//                double3 pixelRGB = make_double3(pixelRGBW.x, pixelRGBW.y, pixelRGBW.z);
//
//                double sigmaR = 2.0;
//                double bilateralWeight = -SquaredLength(pixelRGBW - devPtr[offset]) / (2.0 * sigmaR * sigmaR);
//                double weight = d_kernel[i + KERNEL_RADIUS] * exp(bilateralWeight);
//                //weight = d_kernel[i + KERNEL_RADIUS];
//                sumWeight += weight;
//                result += pixelRGB * weight;
//            }
//        }
//    }
//
//    result /= sumWeight;
//    unsigned char r = static_cast<unsigned char>(result.x);
//    unsigned char g = static_cast<unsigned char>(result.y);
//    unsigned char b = static_cast<unsigned char>(result.z);
//    devBeforeGussian[offset] = make_uchar4(r, g, b, 255);
//    devPtr[offset] = make_uchar4(r, g, b, 255);
//}
__global__ void gaussian(double3* pic, double3* picBeforeGussian, double4* gBuffer, double sigmaG, double sigmaR, double sigmaN, double sigmaD, int max_x, int max_y)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= max_x || y >= max_y) return;

    int offset = y * max_x + x;

    double4 gb = gBuffer[offset];
    double3 normal = make_double3(gb.x, gb.y, gb.z);

    // gaussian
    double3 result = make_double3(0.0, 0.0, 0.0);
    double sumWeight = 0.0;
    for (int i = -KERNEL_RADIUS; i <= KERNEL_RADIUS; i++)
    {
        for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
        {
            int nx = mMin(mMax(x + i, 0), max_x - 1);
            int ny = mMin(mMax(y + j, 0), max_y - 1);

            double3 pixel = picBeforeGussian[ny * max_x + nx];
            double4 gb_p = gBuffer[ny * max_x + nx];
            double3 normal_p = make_double3(gb_p.x, gb_p.y, gb_p.z);

            double gaussianWeight = -(i * i + j * j) / (2.0 * sigmaG * sigmaG);
            double bilateralWeight = -SquaredLength(pixel - pic[offset]) / (2.0 * sigmaR * sigmaR);
            double normalWeight = -SquaredLength(normal - normal_p) / (2.0 * sigmaN * sigmaN);
            double delDepth = gb.w - gb_p.w;
            double depthWeight = -delDepth * delDepth / (2.0 * sigmaD * sigmaD);
            double weight = exp(gaussianWeight + bilateralWeight + normalWeight + depthWeight);
            sumWeight += weight;
            result += pixel * weight;
        }
    }

    result /= sumWeight;
    pic[offset] = result;
}

__global__ void gaussianSeparate(double3* pic, double3* picBeforeGussian, double4* gBuffer, double sigmaG, double sigmaR, double sigmaN, double sigmaD, int max_x, int max_y, bool isHorizontal)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= max_x || y >= max_y) return;

    int offset = y * max_x + x;

    double4 gb = gBuffer[offset];
    double3 normal = make_double3(gb.x, gb.y, gb.z);

    // gaussian
    double3 result = make_double3(0.0, 0.0, 0.0);
    double sumWeight = 0.0;
    for (int i = -KERNEL_RADIUS; i <= KERNEL_RADIUS; i++)
    {
        int nx, ny;
        if (isHorizontal)
        {
            nx = mMin(mMax(x + i, 0), max_x - 1);
            ny = y;
        }
        else
        {
            nx = x;
            ny = mMin(mMax(y + i, 0), max_y - 1);
        }

        double3 pixel = picBeforeGussian[ny * max_x + nx];
        double4 gb_p = gBuffer[ny * max_x + nx];
        double3 normal_p = make_double3(gb_p.x, gb_p.y, gb_p.z);

        double gaussianWeight = -(i * i) / (2.0 * sigmaG * sigmaG);
        double bilateralWeight = -SquaredLength(pixel - pic[offset]) / (2.0 * sigmaR * sigmaR);
        double normalWeight = -SquaredLength(normal - normal_p) / (2.0 * sigmaN * sigmaN);
        double delDepth = gb.w - gb_p.w;
        double depthWeight = -delDepth * delDepth / (2.0 * sigmaD * sigmaD);
        double weight = exp(gaussianWeight + bilateralWeight + normalWeight + depthWeight);
        //weight = kernel[i + KERNEL_RADIUS];
        sumWeight += weight;
        result += pixel * weight;
    }

    result /= sumWeight;
    pic[offset] = result;
}

__global__ void addPrevious(double3* pic, double3* picPrevious, double3* gBufferPosition, const Camera* camera, int max_x, int max_y)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= max_x || y >= max_y) return;

    int offset = y * max_x + x;

    double alpha = 0.1;
    if (camera->isMoving)
        alpha = 0.4;

    double3 position = gBufferPosition[offset];
    int2 ij = camera->getPrePositionInImg(position);
    double3 previous;
    if (ij.x < 0 || ij.y < 0)
        previous = picPrevious[offset]; 
    else
        previous = picPrevious[ij.y * max_x + ij.x];

    pic[offset] = pic[offset] * alpha + previous * (1.0 - alpha);
    //pic[offset] = make_double3(alpha, alpha, alpha);
 
    // Outlier Clamping
    //double3 mean = make_double3(0.0, 0.0, 0.0);
    //double N = 7.0 * 7.0;
    //for (int i = -3; i < 3; i++)
    //{
    //    for (int j = -3; j < 3; j++)
    //    {
    //        int nx = mMin(mMax(x + i, 0), max_x - 1);
    //        int ny = mMin(mMax(y + j, 0), max_y - 1);
    //        mean += picPrevious[ny * max_x + nx];
    //    }
    //}
    //mean /= N;
    //double3 variance = make_double3(0.0, 0.0, 0.0);
    //for (int i = -3; i < 3; i++)
    //{
    //    for (int j = -3; j < 3; j++)
    //    {
    //        int nx = mMin(mMax(x + i, 0), max_x - 1);
    //        int ny = mMin(mMax(y + j, 0), max_y - 1);
    //        double3 pixel = picPrevious[ny * max_x + nx];
    //        variance.x += pow(pixel.x - mean.x, 2);
    //        variance.y += pow(pixel.y - mean.y, 2);
    //        variance.z += pow(pixel.z - mean.z, 2);
    //    }
    //}
    //variance /= N;

    //double3 left = mean - variance;
    //double3 right = mean + variance;
    //double3 left = make_double3(0.0, 0.0, 0.0);
    //double3 right = make_double3(1.0, 1.0, 1.0);

    //previous.x = mMax(left.x, mMin(previous.x, right.x));
    //previous.y = mMax(left.y, mMin(previous.y, right.y));
    //previous.z = mMax(left.z, mMin(previous.z, right.z));
    //pic[offset] = pic[offset] * alpha + previous * (1.0 - alpha);
}

__global__ void pic2RGBW(uchar4* devPtr, double3* pic, int max_x, int max_y)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;

    int offset = j * max_x + i;

    double3 pixel = pic[offset];
    // gamma
    pixel.x = sqrt(pixel.x);
    pixel.y = sqrt(pixel.y);
    pixel.z = sqrt(pixel.z);
    // set color
    unsigned char r = static_cast<unsigned char>(254.99 * CLAMP01(pixel.x));
    unsigned char g = static_cast<unsigned char>(254.99 * CLAMP01(pixel.y));
    unsigned char b = static_cast<unsigned char>(254.99 * CLAMP01(pixel.z));

    devPtr[offset] = make_uchar4(r, g, b, 255);
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