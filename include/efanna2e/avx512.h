#ifndef EFANNA2E_AVX512_H
#define EFANNA2E_AVX512_H
#include <iostream>
#include <cmath>
#include <immintrin.h>

#define VECTOR_SIZE 16 // Number of elements in the vector

float avx512l2translated(
    const float *vec1, const float *vec2, int size)
{
#define AVX512_L2SQR(addr1, addr2, dest, tmp1, tmp2) \
    tmp1 = _mm512_loadu_ps(addr1);                   \
    tmp2 = _mm512_loadu_ps(addr2);                   \
    tmp1 = _mm512_sub_ps(tmp1, tmp2);                \
    tmp1 = _mm512_mul_ps(tmp1, tmp1);                \
    dest = _mm512_add_ps(dest, tmp1);
    __m512 sum;
    __m512 l0, l1;
    __m512 r0, r1;
    unsigned D = (size + 15) & ~15U;
    unsigned DR = D % 32;
    unsigned DD = D - DR;
    const float *l = vec1;
    const float *r = vec2;
    const float *e_l = l + DD;
    const float *e_r = r + DD;
    float unpack[16] __attribute__((aligned(64))) = {0};

    sum = _mm512_loadu_ps(unpack);
    if (DR)
    {
        AVX512_L2SQR(e_l, e_r, sum, l0, r0);
    }

    for (unsigned i = 0; i < DD; i += 32, l += 32, r += 32)
    {
        AVX512_L2SQR(l, r, sum, l0, r0);
        AVX512_L2SQR(l + 16, r + 16, sum, l1, r1);
    }

    _mm512_storeu_ps(unpack, sum);

    // Calculate result
    float result = 0.0f;
    for (int i = 0; i < 16; ++i)
    {
        result += unpack[i];
    }
    return result;
}

// inline
float computeL2Distance(const float *vec1, const float *vec2, int size)
{
    __m512 vsum = _mm512_setzero_ps(); // Initialize sum vector

    // Process 16 elements at a time
    int i;
    for (i = 0; i < size - VECTOR_SIZE; i += VECTOR_SIZE)
    {
        // Load vectors into AVX-512 registers
        __m512 v1 = _mm512_loadu_ps(&vec1[i]);
        __m512 v2 = _mm512_loadu_ps(&vec2[i]);

        // Compute squared differences
        __m512 diff = _mm512_sub_ps(v1, v2);
        __m512 diff_squared = _mm512_mul_ps(diff, diff);

        // Accumulate squared differences
        vsum = _mm512_add_ps(vsum, diff_squared);
    }

    // Handle remaining elements (less than 16)
    float remainingSum = 0.0f;
    for (; i < size; ++i)
    {
        float diff = vec1[i] - vec2[i];
        remainingSum += diff * diff;
    }

    // Horizontal sum of the squared differences
    float temp[VECTOR_SIZE];
    _mm512_storeu_ps(temp, vsum);
    float sum = 0.0f;
    for (int i = 0; i < VECTOR_SIZE; ++i)
    {
        sum += temp[i];
    }
    sum += remainingSum;

    return sum;
}

#endif // EFANNA2E_AVX512_H
