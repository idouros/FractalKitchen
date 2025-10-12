/******************************************************
 * complex.cl — High-performance complex arithmetic
 * Author: Yannis Douros / GPT-5
 * Description:
 *   Optimized complex number operations for OpenCL.
 *   Uses float2 (or double2 if enabled) representation.
 *   Built for fast, numerically stable GPU math.
 ******************************************************/

// Uncomment for double precision (if supported)
// #pragma OPENCL EXTENSION cl_khr_fp64 : enable
// typedef double2 cfloat;
// #define NATIVE_RECIP native_recip
// #define NATIVE_SQRT  native_sqrt
// #define FMA(a,b,c) fma(a,b,c)

typedef float2 cfloat;
#define NATIVE_RECIP native_recip
#define NATIVE_SQRT  native_sqrt
#define FMA(a,b,c) fma(a,b,c)

/*************** BASIC OPERATIONS ****************/

inline cfloat c_add(cfloat a, cfloat b) {
    return a + b; // vectorized addition
}

inline cfloat c_sub(cfloat a, cfloat b) {
    return a - b; // vectorized subtraction
}

inline cfloat c_conj(cfloat a) {
    return (cfloat)(a.x, -a.y);
}

inline cfloat c_neg(cfloat a) {
    return -a;
}

/*************** MULTIPLY / DIVIDE ****************/

inline cfloat c_mul(cfloat a, cfloat b) {
    // (a.x + i*a.y) * (b.x + i*b.y)
    return (cfloat)(
        FMA(a.x, b.x, -a.y*b.y),
        FMA(a.x, b.y,  a.y*b.x)
    );
}

inline cfloat c_div(cfloat a, cfloat b) {
    float denom = FMA(b.x, b.x, b.y*b.y);
    float inv = NATIVE_RECIP(denom);
    return (cfloat)(
        (FMA(a.x, b.x,  a.y*b.y)) * inv,
        (FMA(a.y, b.x, -a.x*b.y)) * inv
    );
}

/*************** MAGNITUDE / PHASE ****************/

inline float c_abs(cfloat a) {
    return NATIVE_SQRT(FMA(a.x, a.x, a.y*a.y));
}

inline float c_arg(cfloat a) {
    return atan2(a.y, a.x);
}

/*************** EXP / LOG / POW ****************/

inline cfloat c_exp(cfloat a) {
    float e = exp(a.x);
    return (cfloat)(e * cos(a.y), e * sin(a.y));
}

inline cfloat c_log(cfloat a) {
    return (cfloat)(log(c_abs(a)), c_arg(a));
}

inline cfloat c_pow(cfloat a, cfloat b) {
    // a^b = exp(b * log(a))
    cfloat ln_a = c_log(a);
    cfloat temp = (cfloat)(
        FMA(b.x, ln_a.x, -b.y * ln_a.y),
        FMA(b.x, ln_a.y,  b.y * ln_a.x)
    );
    return c_exp(temp);
}

/*************** UTILITIES ****************/

inline cfloat c_from_real(float x) {
    return (cfloat)(x, 0.0f);
}

inline float2 c_to_vec(cfloat a) {
    return (float2)(a.x, a.y);
}

/*************** EXAMPLE KERNEL ****************/

__kernel void complex_mul_array(
    __global const cfloat* A,
    __global const cfloat* B,
    __global cfloat* C,
    const int n)
{
    int i = get_global_id(0);
    if (i < n)
        C[i] = c_mul(A[i], B[i]);
}
