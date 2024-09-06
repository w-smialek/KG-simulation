struct hsl{
    double h;    // hue
    double s;    // saturation
    double l;    // Lightness
};

struct rgb{
    double r;   // red
    double g;   // green
    double b;   // blue
};

struct rgba{
    double r;   // red
    double g;   // green
    double b;   // blue
    double a;   // alpha
};


// METHOD 1 (USING POINTER)
// Convert RGB color model into HSL and reciprocally
double * rgb_to_hsl(double r, double g, double b);
double * hsl_to_rgb(double h, double s, double l);
double hue_to_rgb(double m1, double m2, double h);

// METHOD 2 (USING STRUCT)
struct hsl struct_rgb_to_hsl(double r, double g, double b);
struct rgb struct_hsl_to_rgb(double h, double s, double l);

// DETERMINE MAX & MIN VALUES FROM A PIXEL DEFINE WITH RGB VALUES
double fmax_rgb_value(double red, double green, double blue);
double fmin_rgb_value(double red, double green, double blue);
