---
layout: post
title: "A fast implementation of Cosine Similarity with C++ and SIMD"
date: 2023-09-04 00:00:00 -0000
categories: misc
---

Not much to see here other than me putting some of my old and rusty knowledge to work. Knowledge that I gained some years ago in my architecture class to parallelize computations at CPU level, i.e. SIMD, and implement something with it in C++. At the same time I wanted to compare it with Python, the sometimes hated language for high performance lovers[^1].

There’s not much to see here except me dusting off some old knowledge from my architecture class years ago to parallelize computations at the CPU level using SIMD and implement something in C++. At the same time I wanted to compare it with Python, the sometimes hated language for high performance lovers[^1].

Which to me meant implementing a vectorized version of the cosine similarity in C++ and see how it compares to NumPy, SciPy and plain C++. What I implemented looks like below which, besides SciPy, probably not optimal:

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

| Implemntation                       | Time (ms) |
| ----------------------------------- | --------- |
| Numpy                               | 0.6999    |
| Scipy (from the library)            | 0.9893    |
| C++                                 | 2.6558    |
| Vectorized C++                      | 0.5319    |
| C++ with O3 optimization            | 0.4251    |
| Vectorized C++ with O3 optimization | 0.0893    |

| Implementation                      | Time (ms) |
| ----------------------------------- | --------- |
| C++                                 | 2.3839    |
| Numpy                               | 0.6777    |
| C++ with O3 optimization            | 0.5416    |
| Scipy                               | 0.5212    |
| Vectorized C++                      | 0.4694    |
| Vectorized C++ with O3 optimization | 0.0936    |

The complete code for this can be found [here](https://github.com/joseprupi/cosine-similarity-comparison), and to be clear all executions use the same 64000 vectors read from a text file previously generated. Nothing special here, I warned you. Well, Python seems to be comparable to a plain C++ implementation which doesn't seems bad to me, and the C++ vectorized version is 1 order of magnitude faster. So unless you chose the route of implementing a calculation specific for your processor in C++, the C++ libraries that Python is using are decently fast for this task.

But hold on, SciPy is slightly faster than my NumPy implementation and SciPy is implemented using NumPy. I don't think there can be much fantasy in the cosine similarity implementation so let me take a look at it [here](https://github.com/scipy/scipy/blob/main/scipy/spatial/distance.py#L575):

```python 
# Scipy cosine distance implementation

def correlation(u, v, w=None, centered=True):
    """
    Compute the correlation distance between two 1-D arrays.

    The correlation distance between `u` and `v`, is
    defined as

    .. math::

        1 - \\frac{(u - \\bar{u}) \\cdot (v - \\bar{v})}
                  {{\\|(u - \\bar{u})\\|}_2 {\\|(v - \\bar{v})\\|}_2}

    where :math:`\\bar{u}` is the mean of the elements of `u`
    and :math:`x \\cdot y` is the dot product of :math:`x` and :math:`y`.

    Parameters
    ----------
    u : (N,) array_like
        Input array.
    v : (N,) array_like
        Input array.
    w : (N,) array_like, optional
        The weights for each value in `u` and `v`. Default is None,
        which gives each value a weight of 1.0
    centered : bool, optional
        If True, `u` and `v` will be centered. Default is True.

    Returns
    -------
    correlation : double
        The correlation distance between 1-D array `u` and `v`.

    Examples
    --------
    Find the correlation between two arrays.

    >>> from scipy.spatial.distance import correlation
    >>> correlation([1, 0, 1], [1, 1, 0])
    1.5

    Using a weighting array, the correlation can be calculated as:

    >>> correlation([1, 0, 1], [1, 1, 0], w=[0.9, 0.1, 0.1])
    1.1

    If centering is not needed, the correlation can be calculated as:

    >>> correlation([1, 0, 1], [1, 1, 0], centered=False)
    0.5
    """
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
    """
    Compute the Cosine distance between 1-D arrays.

    The Cosine distance between `u` and `v`, is defined as

    .. math::

        1 - \\frac{u \\cdot v}
                  {\\|u\\|_2 \\|v\\|_2}.

    where :math:`u \\cdot v` is the dot product of :math:`u` and
    :math:`v`.

    Parameters
    ----------
    u : (N,) array_like
        Input array.
    v : (N,) array_like
        Input array.
    w : (N,) array_like, optional
        The weights for each value in `u` and `v`. Default is None,
        which gives each value a weight of 1.0

    Returns
    -------
    cosine : double
        The Cosine distance between vectors `u` and `v`.

    Examples
    --------
    >>> from scipy.spatial import distance
    >>> distance.cosine([1, 0, 0], [0, 1, 0])
    1.0
    >>> distance.cosine([100, 0, 0], [0, 1, 0])
    1.0
    >>> distance.cosine([1, 1, 0], [0, 1, 0])
    0.29289321881345254

    """
    # cosine distance is also referred to as 'uncentered correlation',
    #   or 'reflective correlation'
    return correlation(u, v, w=w, centered=False)
```

Ok, and getting rid of all the checks and generalizations of the *correlation* function to make it usable for my use case makes it look like below.

```python 
def cosine(u, v):
    
    uv = np.dot(u, v)
    uu = np.dot(u, u)
    vv = np.dot(v, v)
    dist = 1.0 - uv / math.sqrt(uu * vv)
    # Clip the result to avoid rounding error
    return np.clip(dist, 0.0, 2.0)
```

So clearly what has to make it faster is the denominator, my implementation is using the norm function from NumPy, SciPy uses two NumPy dot multiplications and the Python square root function. Ok, lets run it to see how getting rid of those checks makes it even faster. 0.7 ms, oh crap, this is slower than executing the SciPy version.

Well, it turns out that some of the stuff I got rid off are the couple of lines below:

```python 
u = _validate_vector(u)
v = _validate_vector(v)
```

And _validate_vector is implemented as:

```python 
def _validate_vector(u, dtype=None):
    u = np.asarray(u, dtype=dtype, order='c')
    if u.ndim == 1:
        return u
    raise ValueError("Input vector should be 1-D.")
```
Which not only validates that the input is a vector but also makes the array contigous in memory, something than can't be assured when lo onSo I guess that after reading the vector from the file it is not loaded in contigous blocks of memory and 

[^1]: High Performance Lovers: People that have strong opinions about performance of programming languages, regardless if they ever had to care about it. They know that Python (cPython) is implemented in C and lots of its libraries in C++, and they will let everybody know if Python ever performs reasonably good.