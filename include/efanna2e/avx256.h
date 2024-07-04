#ifndef EFANNA2E_AVX256_H
#define EFANNA2E_AVX256_H

#include <iostream>
#include <cmath>
#include <immintrin.h>

inline float avx256l2translated(
    const float *vec1, const float *vec2, int size)
{
#define AVX_L2SQR(addr1, addr2, dest, tmp1, tmp2) \
    tmp1 = _mm256_loadu_ps(addr1);                \
    tmp2 = _mm256_loadu_ps(addr2);                \
    tmp1 = _mm256_sub_ps(tmp1, tmp2);             \
    tmp1 = _mm256_mul_ps(tmp1, tmp1);             \
    dest = _mm256_add_ps(dest, tmp1);

    __m256 sum;
    __m256 l0, l1;
    __m256 r0, r1;
    unsigned D = (size + 7) & ~7U;
    unsigned DR = D % 16;
    unsigned DD = D - DR;
    const float *l = vec1;
    const float *r = vec2;
    const float *e_l = l + DD;
    const float *e_r = r + DD;
    float unpack[8] __attribute__((aligned(32))) = {0, 0, 0, 0, 0, 0, 0, 0};

    sum = _mm256_loadu_ps(unpack);
    if (DR)
    {
        AVX_L2SQR(e_l, e_r, sum, l0, r0);
    }

    for (unsigned i = 0; i < DD; i += 16, l += 16, r += 16)
    {
        AVX_L2SQR(l, r, sum, l0, r0);
        AVX_L2SQR(l + 8, r + 8, sum, l1, r1);
    }
    _mm256_storeu_ps(unpack, sum);
    float
        result = unpack[0] + unpack[1] + unpack[2] + unpack[3] + unpack[4] + unpack[5] + unpack[6] + unpack[7];
    return result;
}
#endif // EFANNA2E_AVX256_H