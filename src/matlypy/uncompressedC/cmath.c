#define EXPORT __declspec(dllexport)
#define PI 3.14159265358979323846
#define DBL_EPSILON 1e-15

static double c_sqrt(double x) {
    if (x <= 0) return 0;
    double res = x;
    for (int i = 0; i < 10; i++) res = 0.5 * (res + x / res);
    return res;
}

static double c_exp(double x) {
    double sum = 1.0, term = 1.0;
    for (int i = 1; i < 20; i++) {
        term *= x / i;
        sum += term;
    }
    return sum;
}

static double c_acos(double x) {
    double negate = (double)(x < 0);
    if (x < 0) x = -x;
    double ret = -0.0187293;
    ret = ret * x + 0.0742610;
    ret = ret * x - 0.2121144;
    ret = ret * x + 1.5707288;
    ret = ret * c_sqrt(1.0 - x);
    ret = ret - 2 * negate * ret;
    return negate * PI + ret;
}

EXPORT void standardize(const double* in, double* out, int size) {
    double sum = 0, sq_diff_sum = 0;
    for (int i = 0; i < size; i++) sum += in[i];
    double mean = sum / size;
    for (int i = 0; i < size; i++) {
        double d = in[i] - mean;
        sq_diff_sum += d * d;
    }
    double std = c_sqrt(sq_diff_sum / size);
    for (int i = 0; i < size; i++) {
        out[i] = (std < DBL_EPSILON) ? (in[i] - mean) : (in[i] - mean) / std;
    }
}

EXPORT void relu(const double* in, double* out, int size) {
    for (int i = 0; i < size; i++) out[i] = (in[i] > 0) ? in[i] : 0;
}

EXPORT void softmax(const double* in, double* out, int size) {
    double max_v = -1e308;
    for (int i = 0; i < size; i++) if (in[i] > max_v) max_v = in[i];
    double sum = 0;
    for (int i = 0; i < size; i++) {
        out[i] = c_exp(in[i] - max_v);
        sum += out[i];
    }
    for (int i = 0; i < size; i++) out[i] /= sum;
}

EXPORT double tensangle(const double* a, const double* b, int size) {
    double dot = 0, nA = 0, nB = 0;
    for (int i = 0; i < size; i++) {
        dot += a[i] * b[i];
        nA += a[i] * a[i];
        nB += b[i] * b[i];
    }
    double norms = c_sqrt(nA) * c_sqrt(nB);
    if (norms < DBL_EPSILON) return 0.0;
    double cos_t = dot / norms;
    if (cos_t > 1.0) cos_t = 1.0; if (cos_t < -1.0) cos_t = -1.0;
    return c_acos(cos_t) * (180.0 / PI);
}