---
layout: post
title: "A fast implementation of Cosine Similarity with C++ and SIMD"
date: 2023-09-04 00:00:00 -0000
categories: misc
---

Today I wanted to try how to implement a fast version of cosine similarity with C++ and SIMD parallelization. Why? Just to waste some time and learn something. I have no idea about high performance computing so I guess this might seem simple and probably far from optimal.  

Besides implementing a vectorized version of the cosine similarity I wanted to see how this compares to numpy, scipy and plain C++. The different implementations look like:

```python 
# Scipy
cos_sim = distance.cosine(A,B)
```

```python 
# Numpy
cos_sim = np.dot(A, B)/(norm(A)*norm(B)))
```

```C++
// Plain C++
float cosine_similarity(float *A, float *B)
{
    float dot = 0.0, denom_a = 0.0, denom_b = 0.0;
    for (auto i = 0; i < SIZE; ++i)
    {
        dot += A[i] * B[i];
        denom_a += A[i] * A[i];
        denom_b += B[i] * B[i];
    }
    return dot / (sqrt(denom_a) * sqrt(denom_b));
}
```

```C++
// Vectorized C++
inline float simd_horizontal_sum(__m256 &r)
{
    __m128 r4 = _mm_add_ps(_mm256_castps256_ps128(r), _mm256_extractf128_ps(r, 1));
    __m128 r2 = _mm_add_ps(r4, _mm_movehl_ps(r4, r4));
    __m128 r1 = _mm_add_ss(r2, _mm_movehdup_ps(r2));
    return _mm_cvtss_f32(r1);
}

float cosine_similarity_simd(float *A, float *B)
{

    __m256 sum_dot = _mm256_setzero_ps();
    __m256 sum_A = _mm256_setzero_ps();
    __m256 sum_B = _mm256_setzero_ps();

    for (size_t i = 0; i < SIZE; i += 8)
    {
        __m256 buf1 = _mm256_loadu_ps(A + i);
        __m256 buf2 = _mm256_loadu_ps(B + i);

        sum_dot = _mm256_fmadd_ps(buf1, buf2, sum_dot);
        sum_A = _mm256_fmadd_ps(buf1, buf1, sum_A);
        sum_B = _mm256_fmadd_ps(buf2, buf2, sum_B);
    }

    float float_dot = simd_horizontal_sum(sum_dot);
    float float_A_norm = simd_horizontal_sum(sum_A);
    float float_B_norm = simd_horizontal_sum(sum_B);

    return float_dot / (sqrt(float_A_norm) * sqrt(float_B_norm));
}
```

And the average times to calculate the similarity for 2 random vectors of 640000 floats:

| Implemntation | Time (ms) |
| --- | ----------- |
| Numpy | 0.6999 |
| Scipy (from the library) | 0.9893 |
| C++ | 2.6558 |
| Vectorized C++ | 0.5319 |
| C++ with O3 optimization | 0.4251 |
| Vectorized C++ with O3 optimization | 0.0893 |