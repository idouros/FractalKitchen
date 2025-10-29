/******************************************************
 * complex.cl - High-performance complex arithmetic
 * Author: Yannis Douros / GPT-5
 * Description:
 *   Optimized complex number operations for OpenCL.
 *   Uses double-double representation.
 *   Built for precise, numerically stable GPU math.
 ******************************************************/

// Double precision by default if supported, otherwise fall back on single precision
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

typedef struct {
    double hi;
    double lo;
} dd_real;



#define FMA(a,b,c) fma(a,b,c)

// double-double essentials

inline dd_real dd_from_double(double a) {
    dd_real res;
    res.hi = a;
    res.lo = 0.0;
    return res;
}

inline void two_sum(double a, double b, double *sum, double *err) {
    *sum = a + b;
    double bb = *sum - a;
    *err = (a - (*sum - bb)) + (b - bb);
}

inline void two_prod(double a, double b, double *prod, double *err) {
    *prod = a * b;
#if defined(CL_KHR_FP64)
    double a_hi = 0.0, a_lo = 0.0, b_hi = 0.0, b_lo = 0.0;
    const double split = 134217729.0; // 2^27 + 1
    double t = split * a;
    a_hi = t - (t - a);
    a_lo = a - a_hi;
    t = split * b;
    b_hi = t - (t - b);
    b_lo = b - b_hi;
    *err = ((a_hi * b_hi - *prod) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo;
#else
    *err = 0.0; // fallback if splitting not supported
#endif
}

// double-double arithmetic

inline dd_real dd_add(dd_real x, dd_real y) {
    dd_real res;
    double s, e;
    two_sum(x.hi, y.hi, &s, &e);
    e += x.lo + y.lo;
    two_sum(s, e, &res.hi, &res.lo);
    return res;
}

inline dd_real dd_mul(dd_real x, dd_real y) {
    dd_real res;
    double p, e;
    two_prod(x.hi, y.hi, &p, &e);
    e += x.hi * y.lo + x.lo * y.hi;
    two_sum(p, e, &res.hi, &res.lo);
    return res;
}


inline dd_real dd_div(dd_real a, dd_real b) {
    // Step 1: approximate division using high parts only
    double q1 = a.hi / b.hi;

    // Step 2: compute the remainder: r = a - q1 * b
    dd_real qb = dd_mul(dd_from_double(q1), b);
    dd_real r;
    r.hi = a.hi - qb.hi;
    r.lo = a.lo - qb.lo;

    // Step 3: correction term
    double q2 = (r.hi + r.lo) / b.hi;

    // Step 4: sum q1 and q2 using double-double addition
    dd_real res = dd_add(dd_from_double(q1), dd_from_double(q2));

    return res;
}


inline dd_real dd_sqrt(dd_real a) {
    // Approximate sqrt for dd_real using one Newton refinement
    double approx = sqrt(a.hi);
    dd_real x = dd_from_double(approx);
    // Newton iteration: x = 0.5 * (x + a/x)
    dd_real ax = dd_div(a, x);
    dd_real sum = dd_add(x, ax);
    dd_real dd_half = dd_from_double(0.5);
    return dd_mul(sum, dd_half);
}


// complex arithmetic

typedef struct {
    dd_real real;
    dd_real imag;
} dd_complex;

inline dd_complex dd_complex_from_dd(dd_real real_part, dd_real imag_part) {
    dd_complex res;
    res.real = real_part;
    res.imag = imag_part;
    return res;
}

inline dd_complex dd_cadd(dd_complex x, dd_complex y) {
    dd_complex res;
    res.real = dd_add(x.real, y.real);
    res.imag = dd_add(x.imag, y.imag);
    return res;
}

inline dd_complex dd_cmul(dd_complex x, dd_complex y) {
    dd_complex res;
    dd_real ac = dd_mul(x.real, y.real);
    dd_real bd = dd_mul(x.imag, y.imag);
    dd_real ad = dd_mul(x.real, y.imag);
    dd_real bc = dd_mul(x.imag, y.real);

    res.real = dd_add(ac, (dd_real){-bd.hi, -bd.lo});  // ac - bd
    res.imag = dd_add(ad, bc);                          // ad + bc
    return res;
}

inline dd_real dd_cabs(dd_complex z) {
    // z = a + ib
    dd_real a2 = dd_mul(z.real, z.real);   // a^2
    dd_real b2 = dd_mul(z.imag, z.imag);   // b^2
    dd_real sum = dd_add(a2, b2);          // a^2 + b^2
    dd_real mag = dd_sqrt(sum);            // sqrt(a^2 + b^2)
    return mag;
}


inline dd_complex dd_csqrt(dd_complex z) {
    dd_real a = z.real;
    dd_real b = z.imag;

    // Compute |z| = sqrt(a^2 + b^2)
    dd_real a2 = dd_mul(a, a);
    dd_real b2 = dd_mul(b, b);
    dd_real sum = dd_add(a2, b2);
    dd_real mag = dd_sqrt(sum);

    // t1 = (|z| + a) / 2
    dd_real t1 = dd_add(mag, a);
    dd_real dd_half = dd_from_double(0.5);
    t1 = dd_mul(t1, dd_half);

    // t2 = (|z| - a) / 2
    dd_real t2 = dd_add(mag, (dd_real){-a.hi, -a.lo});
    t2 = dd_mul(t2, dd_half);

    // real = sqrt(t1)
    // imag = sign(b) * sqrt(t2)
    dd_real real_part = dd_sqrt(t1);
    dd_real imag_part = dd_sqrt(t2);

    if (b.hi < 0.0 || (b.hi == 0.0 && b.lo < 0.0)) {
        imag_part.hi = -imag_part.hi;
        imag_part.lo = -imag_part.lo;
    }

    dd_complex res;
    res.real = real_part;
    res.imag = imag_part;
    return res;
}