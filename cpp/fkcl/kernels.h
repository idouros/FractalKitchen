#ifndef _KERNELS
#define _KERNELS


const char* kernelMandelbrot = R"CLC(
__kernel void init_image(write_only image2d_t img, float x_start, float y_start, float pixel_step, unsigned int max_iter) {
    int col = get_global_id(0);
    int row = get_global_id(1);

    double y = y_start + (float)row * pixel_step;
    double x = x_start + (float)col * pixel_step;

    float ret = 0.0f;
    unsigned int i = 0;
    bool keep_going = true;
    cfloat z = (cfloat)(0.0f, 0.0f); // TODO: Make user configurable
    cfloat c = (cfloat)(x, y);
    while(keep_going)
    {
        z = c_add(c_mul(z, z), c);
        i += 1;
        if( (i >= max_iter) || (c_abs(z) > 2) ) // TODO: Make user configurable
        {
            keep_going = false;
        }
    }
    if(c_abs(z) > 2)
    {
        ret = (float)(max_iter - i) / (float)max_iter;
    }

    // float4 is required for image writes
    float4 pixel = (float4)(ret, 0.0f, 0.0f, 0.0f);
    write_imagef(img, (int2)(col, row), pixel);
}
)CLC";


const char* kernelJulia = R"CLC(
__kernel void init_image(write_only image2d_t img, float x_start, float y_start, float pixel_step, unsigned int max_iter) {
    int col = get_global_id(0);
    int row = get_global_id(1);

    double y = y_start + (float)row * pixel_step;
    double x = x_start + (float)col * pixel_step;

    float ret = 0.0f;
    unsigned int i = 0;
    bool keep_going = true;
    cfloat c = (cfloat)(-0.8f, 0.156f);  // TODO: Make user configurable
    cfloat z = (cfloat)(x, y);
    while(keep_going)
    {
        z = c_add(c_mul(z, z), c);
        i += 1;
        if( (i >= max_iter) || (c_abs(z) > 2) ) // TODO: Make user configurable
        {
            keep_going = false;
        }
    }
    if(c_abs(z) > 2)
    {
        ret = (float)(max_iter - i) / (float)max_iter;
    }

    // float4 is required for image writes
    float4 pixel = (float4)(ret, 0.0f, 0.0f, 0.0f);
    write_imagef(img, (int2)(col, row), pixel);
}
)CLC";


// This is to enable specifying which kernel to use in a config file
std::unordered_map<std::string, std::string> kernels = {
    {"Mandelbrot", kernelMandelbrot},
    {"Julia", kernelJulia},
};

#endif

