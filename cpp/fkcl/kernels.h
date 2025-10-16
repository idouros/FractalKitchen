const char* kernelMandelbrot = R"CLC(
__kernel void init_image(write_only image2d_t img, float x_start, float y_start, float pixel_step) {
    int col = get_global_id(0);
    int row = get_global_id(1);

    float y = y_start + (float)row * pixel_step;
    float x = x_start + (float)col * pixel_step;

    float ret = 0.0f;
    unsigned int max_iter = 90;
    unsigned int i = 0;
    bool keep_going = true;
    cfloat z = (cfloat)(0.0f, 0.0f);
    cfloat c = (cfloat)(x, y);
    while(keep_going)
    {
        z = c_add(c_mul(z, z), c);
        i += 1;
        if( (i >= max_iter) || (c_abs(z) > 2) )
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
__kernel void init_image(write_only image2d_t img, float x_start, float y_start, float pixel_step) {
    int col = get_global_id(0);
    int row = get_global_id(1);

    float y = y_start + (float)row * pixel_step;
    float x = x_start + (float)col * pixel_step;

    float ret = 0.0f;
    unsigned int max_iter = 900;
    unsigned int i = 0;
    bool keep_going = true;
    cfloat c = (cfloat)(-0.8f, 0.156f);
    cfloat z = (cfloat)(x, y);
    while(keep_going)
    {
        z = c_add(c_mul(z, z), c);
        i += 1;
        if( (i >= max_iter) || (c_abs(z) > 2) )
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