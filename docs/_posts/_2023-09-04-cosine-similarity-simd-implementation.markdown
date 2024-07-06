---
layout: post
title: "A fast implementation of Cosine Similarity with C++ and SIMD"
date: 2023-09-04 00:00:00 -0000
categories: misc
---

There’s not much to see here except me dusting off some old knowledge from my architecture class years ago to parallelize computations using SIMD and implement something in C++. At the same time, I wanted to compare it with Python, a sometimes hated language among pointer and template lovers.[^1].

This in my mind crystallized in implementing a vectorized version of the cosine similarity in C++ to see how it compares to NumPy, SciPy, and plain C++ and writing a shiny post about the results. 

What I implemented and the average times to calculate the similarity for 2 random vectors of 640000 floats are below, which, besides SciPy, is probably far from optimal. 

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

| Implementation                      | Time (ms) |
| ----------------------------------- | --------- |
| C++                                 | 2.3839    |
| Numpy                               | 0.6777    |
| C++ with O3 optimization            | 0.5416    |
| Scipy                               | 0.5212    |
| Vectorized C++                      | 0.4694    |
| Vectorized C++ with O3 optimization | 0.0936    |

The complete implementation can be found [here](https://github.com/joseprupi/cosine-similarity-comparison). 

Aaaand... nothing special to see, I warned you. SIMD is fast and Python is slow.

Well, Python libraries appear to be comparable to a plain C++ implementation, which isn't bad in my opinion. The vectorized C++ version, however, is an order of magnitude faster. Therefore, unless you opt to implement a processor-specific calculation in C++, the C++ libraries that Python uses are decently fast for this task.

But wait, SciPy is slightly faster than my NumPy implementation even though SciPy is built using NumPy, and there shouldn't be much fantasy in the cosine similarity implementation, so let me check it out [here](https://github.com/scipy/scipy/blob/main/scipy/spatial/distance.py#L575):

```python 
def correlation(u, v, w=None, centered=True):
    
    u = _validate_vector(u)
    v = _validate_vector(v)
    if w is not None:
        w = _validate_weights(w)
        w = w / w.sum()
    if centered:
        if w is not None:
            umu = np.dot(u, w)
            vmu = np.dot(v, w)
        else:
            umu = np.mean(u)
            vmu = np.mean(v)
        u = u - umu
        v = v - vmu
    if w is not None:
        vw = v * w
        uw = u * w
    else:
        vw, uw = v, u
    uv = np.dot(u, vw)
    uu = np.dot(u, uw)
    vv = np.dot(v, vw)
    dist = 1.0 - uv / math.sqrt(uu * vv)
    # Clip the result to avoid rounding error
    return np.clip(dist, 0.0, 2.0)

def cosine(u, v, w=None):
    return correlation(u, v, w=w, centered=False)
```

First, let's simplify the correlation function by removing unnecessary checks and generalizations for my use case which will make it more readable and likely faster:

```python 
def cosine(u, v):
    
    uv = np.dot(u, v)
    uu = np.dot(u, u)
    vv = np.dot(v, v)
    dist = 1.0 - uv / math.sqrt(uu * vv)
    
    return np.clip(dist, 0.0, 2.0)
```

My initial thought is that the denominator is making it faster, I have no idea why but that is the only difference I see. My implementation is using the *norm* function from NumPy while SciPy uses two NumPy dot multiplications and the Python *sqrt* function. Let’s run it to ensure everything works as expected and also see how removing those checks improves performance. 

And the result is 0.7 ms, which oh crap, is slower than executing the original SciPy version and similar to my NumPy implementation. It turns out that some of the stuff I got rid off from the *correlation* function are the two lines below: 

```python 
u = _validate_vector(u)
v = _validate_vector(v)
```
 
The name seems pretty explicit, validate the vectors, something I don't need as I know what I am using at runtime. But *_validate_vector* is implemented as:

```python 
def _validate_vector(u, dtype=None):
    u = np.asarray(u, dtype=dtype, order='c')
    if u.ndim == 1:
        return u
    raise ValueError("Input vector should be 1-D.")
```
which not only validates that the input is a vector but also makes the array contiguous in memory calling *np.asarray*, something that can't be assured when loading a vector from disk as I am doing. Something needed for 

So, making the vector contiguous in memory and then perform the calculations seems to be worth it in terms of time as it is slightly faster than not doing so. Which makes sense as I am guessing that the underlying libraries can take advantage of.

Just to be sure this is the case let's reload the vectors, make the contiguous and see if my NumPy and SciPy are close.

```python 
file_data = np.genfromtxt('../tools/vectors.csv', delimiter=',')
A,B = np.moveaxis(file_data, 1, 0).astype('f')

A = np.asarray(A, dtype='float', order='c')
B = np.asarray(B, dtype='float', order='c')

accum = 0

for _ in range(EXECUTIONS):
    start_time = time.time()
    cos_sim = 1 - np.dot(A, B)/(norm(A)*norm(B))
    accum +=  (time.time() - start_time) * 1000

print(" %s ms" % (accum/EXECUTIONS))

def cosine(u, v):
    
    uv = np.dot(u, v)
    uu = np.dot(u, u)
    vv = np.dot(v, v)
    dist = 1.0 - uv / math.sqrt(uu * vv)
    # Clip the result to avoid rounding error
    return np.clip(dist, 0.0, 2.0)

accum = 0

for _ in range(EXECUTIONS):
    start_time = time.time()
    cos_sim = cosine(A,B)
    accum += (time.time() - start_time) * 1000

print(" %s ms" % (accum/EXECUTIONS))

```

The results are **0.04539 ms** and **0.04254 ms** which confirms they are similar, but, mmmmm... not only that they are faster than my SIMD implementation, x2 faster.

[^1]: High Performance Lovers: People that have strong opinions about performance of programming languages, regardless if they ever had to care about it. They know that Python (cPython) is implemented in C and lots of its libraries in C++, and they will let everybody know if Python ever performs reasonably good.