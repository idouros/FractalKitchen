
#include <corecrt_math_defines.h>

// TODO: map enums to strings and use in config files
// also: the colour cycles
// and map the keystrokes

enum ColourMode
{
    COLOUR_MODE_HSV = 0,
    COLOUR_MODE_BBCW,
    COLOUR_MODE_FLAME
};

inline uint8_t clamp255(double x)
{
    return static_cast<uint8_t>(std::max(0.0, std::min(255.0, x)));
}

// Cosine interpolation between two values
inline double coslerp(double a, double b, double t)
{
    double ft = t * M_PI;
    double f = (1 - std::cos(ft)) * 0.5;
    return a*(1-f) + b*f;
}



inline cv::Vec3b smoothFlame(const double val0, const double cycles = 1.0)
{
    auto val = std::clamp(val0, 0.0, 1.0);
    val = std::fmod(val * cycles, 1.0);
    cv::Vec3b colour; // B, G, R
    double gamma = 0.5;
    double t;

    if (val < 0.33)
    {
        // Black → Red
        auto t = std::pow(val / 0.33, gamma);
        colour[0] = 0;                                       // B
        colour[1] = 0;                                       // G
        colour[2] = static_cast<uint8_t>(t / 0.33 * 255.0);  // R
    }
    else if (val < 0.66)
    {
        // Red → Orange → Yellow
        t = std::pow((val - 0.33) / 0.33, gamma);
        colour[0] = 0;                                       // B
        colour[1] = static_cast<uint8_t>(t * 255.0);           // G
        colour[2] = 255;                                     // R
    }
    else
    {
        // Yellow → White
        t = std::pow((val - 0.66) / 0.34, gamma);
        colour[0] = static_cast<uint8_t>(t * 255.0);           // B
        colour[1] = 255;                                     // G
        colour[2] = 255;                                     // R
    }
    return colour;
}

// Smooth Black → Blue → Cyan → White
inline cv::Vec3b smoothBBCW(const double val0, const double cycles = 1.0)
{
    auto val = std::clamp(val0, 0.0, 1.0);
    val = std::fmod(val * cycles, 1.0);
    cv::Vec3b colour; // B, G, R

    if (val < 0.33)
    {
        // Black → Blue
        double t = val / 0.33;
        colour[0] = clamp255(coslerp(0, 255, t));    // B
        colour[1] = 0;                               // G
        colour[2] = 0;                               // R
    }
    else if (val < 0.66)
    {
        // Blue → Cyan
        double t = (val - 0.33) / 0.33;
        colour[0] = 255;                             // B
        colour[1] = clamp255(coslerp(0, 255, t));    // G
        colour[2] = 0;                               // R
    }
    else
    {
        // Cyan → White
        double t = (val - 0.66) / 0.34;
        colour[0] = 255;                             // B
        colour[1] = 255;                             // G
        colour[2] = clamp255(coslerp(0, 255, t));    // R
    }
    return colour;
}

// TO DO: Introduce histogram colouring


inline cv::Vec3b smoothHSV(const double val, const double cycles = 1.0)
{
    cv::Vec3b colour; // H, S, V
    colour[0] = static_cast<int>(std::fmod(val * cycles * 179.0f, 179.0f)); // H
    colour[1] = static_cast<int>(200 + 55 * std::sqrt(val)); // 200–255        // S
    colour[2] = static_cast<int>(std::lround(255.0f * std::pow(val, 0.3f)));   // V
    return colour; 
}
