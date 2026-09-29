
#include <corecrt_math_defines.h>
#include <map>

#define DEFAULT_COLOUR_MODE "HSV"

#define CREATE_ENUM(name) name,
#define CREATE_MAP(S, ...) { #S, ENUM_CLASS_NAME::##S },

#define COLOUR_MODES(colour_mode) \
	colour_mode(HSV)\
	colour_mode(BBCW)\
	colour_mode(FLAME)\
    colour_mode(DISTANCE_CONTOURS)
#define ENUM_CLASS_NAME ColourMode 
enum class ENUM_CLASS_NAME { COLOUR_MODES(CREATE_ENUM) };
static std::map<std::string, ENUM_CLASS_NAME> colourModeMap = { COLOUR_MODES(CREATE_MAP) };
#undef ENUM_CLASS_NAME

inline uint8_t clamp255(double x)
{
    return static_cast<uint8_t>(std::max(0.0, std::min(255.0, x)));
}

inline double histRemap(const double val, std::vector<size_t>* cumulativeHistogram = nullptr)
{
    // Optional histogram/CDF remapping
    if (cumulativeHistogram != nullptr &&
        !cumulativeHistogram->empty() &&
        cumulativeHistogram->back() > 0)
    {
        const size_t numBins = cumulativeHistogram->size();
        const double total = static_cast<double>(cumulativeHistogram->back());

        // Continuous position within the histogram
        const double binPos = val * (numBins - 1);

        // Adjacent bins
        const size_t bin0 = static_cast<size_t>(binPos);
        const size_t bin1 = std::min(bin0 + 1, numBins - 1);

        // Fractional position between the two bins
        const double t = binPos - static_cast<double>(bin0);

        // CDF values at the two bins
        const double cdf0 =
            static_cast<double>((*cumulativeHistogram)[bin0]) / total;

        const double cdf1 =
            static_cast<double>((*cumulativeHistogram)[bin1]) / total;

        // Linear interpolation between CDF values
        return cdf0 + t * (cdf1 - cdf0);
    }
    else 
    {
        return val;
    }
}

// Cosine interpolation between two values
inline double coslerp(double a, double b, double t)
{
    double ft = t * M_PI;
    double f = (1 - std::cos(ft)) * 0.5;
    return a*(1-f) + b*f;
}

inline cv::Vec3b smoothFlame(const double val0, const double cycles = 1.0, std::vector<size_t>* cumulativeHistogram = nullptr)
{
    auto val = std::clamp(val0, 0.0, 1.0);
    val = histRemap(val, cumulativeHistogram);   
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
        colour[1] = static_cast<uint8_t>(t * 255.0);         // G
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

// Smooth Black → Blue → Cyan → White*
inline cv::Vec3b smoothBBCW(const double val0, const double cycles = 1.0, std::vector<size_t>* cumulativeHistogram = nullptr)
{
    auto val = std::clamp(val0, 0.0, 1.0);
  
     // Mandelbrot interior
    if (val <= 0.0)
        return cv::Vec3b(0, 0, 0);

    val = histRemap(val, cumulativeHistogram);    
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

inline cv::Vec3b smoothHSV(const double val0, const double cycles = 1.0, std::vector<size_t>* cumulativeHistogram = nullptr)
{
    auto val = std::clamp(val0, 0.0, 1.0);
    val = histRemap(val, cumulativeHistogram);    

    cv::Vec3b colour; // H, S, V
    colour[0] = static_cast<int>(std::fmod(val * cycles * 179.0f, 179.0f));    // H
    colour[1] = static_cast<int>(200 + 55 * std::sqrt(val)); // 200–255        // S
    colour[2] = static_cast<int>(std::lround(255.0f * std::pow(val, 0.3f)));   // V
    return colour; 
}

inline cv::Vec3b distanceContours(
    const double distance,
    const double pixelStep)
{
    if (distance <= 0.0)
        return cv::Vec3b(0, 0, 0);

    const double d = distance / pixelStep;

    // Log distance gives approximately geometrically spaced contours
    const double x = std::log1p(d);

    // Repeating smooth bands
    const double v = 0.5 + 0.5 * std::cos(x * 10.0);

    const uint8_t c = static_cast<uint8_t>(
        std::clamp(v * 255.0, 0.0, 255.0));

    return cv::Vec3b(c, c, c);
}

