---
layout: post
title: "A not that fast implementation of Cosine Similarity with C++ and SIMD"
date: 2024-07-08 00:00:00 -0000
categories: misc
---

There’s not much to see here except me dusting off some old knowledge from my architecture class years ago to parallelize computations using SIMD, implement something in C++ and compare it with Python.

Which to me meant implementing a vectorized version of the cosine similarity in C++ to see how it compares to Python,  NumPy, SciPy, and plain C++. 

What I implemented and the average times to calculate the similarity for 2 random vectors of 640000 floats are below, which, besides SciPy, is probably far from optimal. 

The details and complete implementation can be found [here](https://github.com/joseprupi/cosine-similarity-comparison).

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

Implementation   |   Time (ms) | 
|-----------------|------------:|
| Vectorized C++ | 0.0936    |
| C++            | 0.5416    |
| SciPy            |    0.5494 | 
| NumPy            |    0.6953 | 
| Plain Python     |  323.389    |  

Aaaand... nothing special to see, I warned you. SIMD is fast and Python is slow.

Well, maybe we can do better with Python and do it in a more pythonic way.

```python 
def cosine(A, B):
    dot = denom_a = denom_b = 0.0

    dot = sum([a*b for a,b in zip(A,B)])
    denom_a = sum([x*x for x in A])
    denom_b = sum([x*x for x in B])

    return 1 - (dot / (math.sqrt(denom_a) * math.sqrt(denom_b)))
```

Implementation   |   Time (ms) | 
|-----------------|------------:|
| Vectorized C++ | 0.0936    |
| C++                                 | 0.5416    |
| SciPy            |    0.5494 | 
| NumPy            |    0.6953 | 
| <span style="color:red">Plain Python, more pythonic</span>    |  <span style="color:red">271.683</span>     |  
| Plain Python     |  323.389    |  

~20% faster than less pythonic Python but still not competing with the other options.

Well, also Python libraries appear to be comparable to a plain C++ implementation, which isn't bad in my opinion. Yes, the vectorized C++ version is an order of magnitude faster, but unless you opt to implement a processor-specific calculation in C++, the libraries that Python uses and people talk about are decently fast for this task.

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

 My initial thought is that the denominator is what is making it faster, I have no idea why but that is the only difference I see. My implementation is using the *norm* function from NumPy while SciPy uses two NumPy dot multiplications and the Python *sqrt* function.

But before keep digging into where the difference is, let's simplify the correlation function by removing unnecessary checks and generalizations for my use case which will make it more readable and likely faster:

```python 
def cosine(u, v):
    
    uv = np.dot(u, v)
    uu = np.dot(u, u)
    vv = np.dot(v, v)
    dist = 1.0 - uv / math.sqrt(uu * vv)
    
    return np.clip(dist, 0.0, 2.0)
```

And now re-run it to ensure everything works as expected and also see how removing those checks improves performance. 

Implementation   |   Time (ms) | 
|-----------------|------------:|
| Vectorized C++ | 0.0936    |
| C++                                 | 0.5416    |
| SciPy            |    0.5494 |
| <span style="color:red">NumPy as Scipy</span>   |    <span style="color:red">0.6714</span> | 
| NumPy            |    0.6953 | 
| Plain Python, more pythonic    |  271.683     |  
| Plain Python     |  323.389    |  

And the result is ~0.7 ms, which oh crap, is slower than executing the original SciPy version and almost identical to my original NumPy implementation. 

It turns out that some of the stuff I got rid off from the *correlation* function are the two lines below: 

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
Which not only validates that the input is a 1xn array but also makes the array contiguous in memory calling *np.asarray* (see *order* parameter from [NumPy documentation](https://numpy.org/doc/stable/reference/generated/numpy.asarray.html)), something that can't be guaranteed when loading a vector from disk as I am doing. 

And as CPUs love working with contiguous things, having things contiguous in memory is always good to go fast, either to compute things in parallel or access things faster from cache, something the underlying Python libraries that people talk about can probably take advantage of.

When looking at the results, making the vector contiguous in memory plus performing the calculations seems to be worth it in this case  as it is slightly faster than not doing so. 

Just to be sure this is the case, let's reload the vectors, make them contiguous and see if my NumPy and SciPy are close.

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

Implementation   |   Time (ms) | 
|-----------------|------------:|
| <span style="color:red">NumPy. Contiguous array</span>   |    <span style="color:red">0.04193<span> | 
| <span style="color:red">NumPy as Scipy. Contiguous array</span>   |    <span style="color:red">0.04264</span> | 
| Vectorized C++ | 0.0936    |
| SciPy            |    0.549369 | 
| NumPy            |    0.695258 | 
| C++              | 2.3839    |
| Plain Python, more pythonic   |  271.683     |  
| Plain Python     |  323.389    |  

The results confirm they are faster when making the arrays contiguous , but, mmmmm... not only that, they are also faster than my SIMD implementation, x2 faster.

Why? Because they are calculated using the BLAS library available in the OS, which means that not even writing C++ SIMD code will make me have a faster implementation than the one Python is using and I will probably have to write my own assembly code with compiler-like tricks to go as fast as Python plus C++ libraries. 

Or I can also use the same libraries Python is using from C++. 

But nah, to me and my needs is enough to not feel guilty when not thinking: "Maybe I should not be using Python as it is slow, and if it is fast is not because of Python, that as someone told me once, is slow, but because it is using C/C++ libraries underneath".

I am a lazy not-good-enough software developer and most of the time I have deadlines to accomplish.