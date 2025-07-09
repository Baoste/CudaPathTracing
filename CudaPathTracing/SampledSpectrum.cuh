#pragma once

#include "rtmath.cuh"

class SampledSpectrum 
{
public:
    double c[SAMPLE_WAVELENGTH] = { 0.0 };

public:
    __host__ __device__  SampledSpectrum()
    {
        for (int i = 0; i < SAMPLE_WAVELENGTH; ++i)
            c[i] = 0.0;
    }
    __host__ __device__  SampledSpectrum(double v)
    {
        for (int i = 0; i < SAMPLE_WAVELENGTH; ++i)
            c[i] = v;
    }
    __host__ __device__  SampledSpectrum(const SampledSpectrum& other)
    {
        for (int i = 0; i < SAMPLE_WAVELENGTH; ++i)
            c[i] = other.c[i];
    }

    __host__ __device__  SampledSpectrum& operator=(const SampledSpectrum& other)
    {
        if (this != &other) 
        {
            for (int i = 0; i < SAMPLE_WAVELENGTH; i++)
                c[i] = other.c[i];
        }
        return *this;
    }
    __host__ __device__  double& operator[](int i)
    {
        return c[i];
    }
    __host__ __device__  const double& operator[](int i) const
    {
        return c[i];
    }

    __host__ __device__  SampledSpectrum& operator*=(double scalar)
    {
        for (int i = 0; i < SAMPLE_WAVELENGTH; i++)
            c[i] *= scalar;
        return *this;
    }
    __host__ __device__  SampledSpectrum& operator*=(const SampledSpectrum& other)
    {
        for (int i = 0; i < SAMPLE_WAVELENGTH; i++)
            c[i] *= other[i];
        return *this;
    }
    __host__ __device__  SampledSpectrum& operator/=(double scalar)
    {
        for (int i = 0; i < SAMPLE_WAVELENGTH; i++)
            c[i] /= scalar;
        return *this;
    }
    __host__ __device__  SampledSpectrum& operator+=(const SampledSpectrum& other)
    {
        for (int i = 0; i < SAMPLE_WAVELENGTH; i++)
            c[i] += other[i];
        return *this;
    }
};