---
layout: post
title: "A not so fast implementation of cosine similarity in C++ and SIMD"
date: 2024-07-08 00:00:00 -0000
categories: misc
---

There isn’t much to see here, just me dusting off some old architecture knowledge to parallelize computations using SIMD, implement it in C++, and compare the results with Python.

The task involved creating a vectorized version of the cosine similarity in C++ to compare its performance against Python, NumPy, SciPy, and plain C++ implementations.

Below is my implementation and the average times for calculating the similarity between two random vectors of 640,000 floats. The full code can be found [here](https://github.com/joseprupi/cosine-similarity-comparison).

```python 
# Python
def cosine(A, B):
    dot = denom_a = denom_b = 0.0

    for i in range(len(A)):
        dot += A[i] * B[i]
        denom_a += A[i] * A[i]
        denom_b += B[i] * B[i]

    return 1 - (dot / (math.sqrt(denom_a) * math.sqrt(denom_b)))
```

```python 
# Scipy
cos_sim = distance.cosine(A,B)
```

```python 
# Numpy
cos_sim = np.dot(A, B)/(norm(A)*norm(B)))
```

```cpp
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

```cpp
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

|Implementation   |   Time (ms) | 
|-----------------|------------:|
| Vectorized C++ | 0.0936    |
| C++            | 0.5416    |
| SciPy            |    0.5494 | 
| NumPy            |    0.6953 | 
| Plain Python     |  323.389    |  

As expected, SIMD is the fastest and plain Python is frustratingly slow. Yes, maybe we can do better with the plain Python implementation making it more "pythonic":

```python 
def cosine(A, B):
    dot = denom_a = denom_b = 0.0

    dot = sum([a*b for a,b in zip(A,B)])
    denom_a = sum([x*x for x in A])
    denom_b = sum([x*x for x in B])

    return 1 - (dot / (math.sqrt(denom_a) * math.sqrt(denom_b)))
```

|Implementation   |   Time (ms) | 
|-----------------|------------:|
| Vectorized C++ | 0.0936    |
| C++                                 | 0.5416    |
| SciPy            |    0.5494 | 
| NumPy            |    0.6953 | 
| <span style="color:red">Plain Python, more pythonic</span>    |  <span style="color:red">271.683</span>     |  
| Plain Python     |  323.389    |  

Which results in a ~20% improvement, but still nowhere near the performance of the other implementations, so I’ll stop here and shift my focus to Python libraries.

Interestingly, Python’s libraries are comparable to plain C++ in performance, which isn't bad in my opinion. Yes, the vectorized C++ version is an order of magnitude faster, but unless you opt to implement a processor-specific calculation in C++, the Python libraries are decently optimized for tasks like this.

However, SciPy is slightly faster than my NumPy implementation even though it is built using NumPy, and there shouldn't be much fantasy in the cosine similarity implementation, so let me check it out [here](https://github.com/scipy/scipy/blob/main/scipy/spatial/distance.py#L575):

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

 My initial thought was that the difference in performance was due to the denominator, I have no idea why but that is the only difference I see. My implementation is using the *norm* function from NumPy while SciPy uses two NumPy dot multiplications and the Python *sqrt* function.

Before diving deeper into the difference, I simplified the correlation function by removing extra checks and generalizations that weren’t needed for my use case, making it easier to read and hopefully a bit faster:

```python 
def cosine(u, v):
    
    uv = np.dot(u, v)
    uu = np.dot(u, u)
    vv = np.dot(v, v)
    dist = 1.0 - uv / math.sqrt(uu * vv)
    
    return np.clip(dist, 0.0, 2.0)
```

And now, re-run it to ensure everything works as expected and also see if removing those checks improves performance.

|Implementation   |   Time (ms) | 
|-----------------|------------:|
| Vectorized C++ | 0.0936    |
| C++                                 | 0.5416    |
| SciPy            |    0.5494 |
| <span style="color:red">NumPy as Scipy</span>   |    <span style="color:red">0.6714</span> | 
| NumPy            |    0.6953 | 
| Plain Python, more pythonic    |  271.683     |  
| Plain Python     |  323.389    |  

And the result is ~0.7 ms, which oh crap, is slower than executing the original SciPy version and almost identical to my original NumPy implementation. It turns out that I removed two important lines from the *correlation* function:

```python 
u = _validate_vector(u)
v = _validate_vector(v)
```
 
The name seems pretty explicit, validate the vectors, something I don't need as I know what I am using at runtime, but *_validate_vector* is implemented as:

```python 
def _validate_vector(u, dtype=None):
    u = np.asarray(u, dtype=dtype, order='c')
    if u.ndim == 1:
        return u
    raise ValueError("Input vector should be 1-D.")
```
Which not only validates that the input is a **1 x n** array but also makes the array contiguous in memory calling *np.asarray* (see *order* parameter from [NumPy documentation](https://numpy.org/doc/stable/reference/generated/numpy.asarray.html)), something that can't be guaranteed when loading a vector from disk as I am doing. 

As CPUs love working with contiguous things, having things contiguous in memory is always good to go fast, either to compute in parallel or access things faster from cache, something the underlying Python libraries can probably take advantage of. So the time spent making vectors contiguous in memory, along with performing the calculations, seems worth it in this case as it is slightly faster.

Just to be sure this is true, let's reload the vectors, make them contiguous and see if my NumPy and SciPy are close in execution time this way.

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

|Implementation   |   Time (ms) | 
|-----------------|------------:|
| <span style="color:red">NumPy. Contiguous array</span>   |    <span style="color:red">0.04193<span> | 
| <span style="color:red">NumPy as Scipy. Contiguous array</span>   |    <span style="color:red">0.04264</span> | 
| Vectorized C++ | 0.0936    |
| SciPy            |    0.549369 | 
| NumPy            |    0.695258 | 
| C++              | 2.3839    |
| Plain Python, more pythonic   |  271.683     |  
| Plain Python     |  323.389    |  

The results show that making arrays contiguous does improve performance significantly. Mmmmm... not only that, they are also faster than my SIMD implementation, twice as fast in fact.

Why? Because they are calculated using the BLAS library available in the OS, which means that not even writing C++ SIMD code will make me have a faster implementation than the one Python is using and I will probably have to write my own assembly code with compiler-like tricks to go as fast as Python plus C++ libraries. 

Honestly Python[^1] is lightning fast for my purposes and deadlines.

[^1]: Where Python is the experience you get when you open VS Code and start coding—no specifics about language implementations, libraries, or interpreters. It's all about diving into coding quickly. If you view Python differently, this might not be for you.