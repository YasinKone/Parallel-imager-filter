
#ifndef IMAGE_FILTERS_H
#define IMAGE_FILTERS_H

#include <algorithm>
#include <cmath>
#include <vector>

struct Image {
    int width = 0;
    int height = 0;
    int channels = 0;
    std::vector<unsigned char> data;
};

static inline unsigned char clamp_u8(float val) {
    if (val < 0.0f) return 0;
    if (val > 255.0f) return 255;
    return static_cast<unsigned char>(std::lround(val));
}

// GRAYSCALE: BASELINE SERIAL
inline Image gray_baseline_serial(const Image& src) {
    Image dst{src.width, src.height, 1,
        std::vector<unsigned char>(
            static_cast<size_t>(src.width) * src.height
        )
    };

    int ch = src.channels;

    for (int y = 0; y < src.height; ++y) {
        for (int x = 0; x < src.width; ++x) {
            size_t idx = (static_cast<size_t>(y) * src.width + x) * ch;

            float r = src.data[idx];
            float g = (ch > 1) ? src.data[idx + 1] : r;
            float b = (ch > 2) ? src.data[idx + 2] : r;

            float gray = 0.299f * r + 0.587f * g + 0.114f * b;

            dst.data[static_cast<size_t>(y) * src.width + x] =
                clamp_u8(gray);
        }
    }

    return dst;
}

// GRAYSCALE: CACHE-FRIENDLY
inline Image gray_cache_friendly(const Image& src) {
    Image dst{src.width, src.height, 1,
        std::vector<unsigned char>(
            static_cast<size_t>(src.width) * src.height
        )
    };

    const int w = src.width;
    const int h = src.height;
    const int ch = src.channels;

    const unsigned char* s = src.data.data();
    unsigned char* d = dst.data.data();

    for (int y = 0; y < h; ++y) {
        const size_t row_offset = static_cast<size_t>(y) * w;
        const unsigned char* row_src = s + row_offset * ch;
        unsigned char* row_dst = d + row_offset;

        if (ch >= 3) {
            for (int x = 0; x < w; ++x) {
                const size_t p = static_cast<size_t>(x) * ch;

                float gray =
                    0.299f * row_src[p] +
                    0.587f * row_src[p + 1] +
                    0.114f * row_src[p + 2];

                row_dst[x] = clamp_u8(gray);
            }
        } else if (ch == 2) {
            for (int x = 0; x < w; ++x) {
                const size_t p = static_cast<size_t>(x) * 2;

                float r = row_src[p];
                float g = row_src[p + 1];

                row_dst[x] =
                    clamp_u8(0.299f * r + 0.587f * g + 0.114f * r);
            }
        } else {
            std::copy(row_src, row_src + w, row_dst);
        }
    }

    return dst;
}

// GRAYSCALE: OPENMP
inline Image gray_openmp(const Image& src) {
    Image dst{src.width, src.height, 1,
        std::vector<unsigned char>(
            static_cast<size_t>(src.width) * src.height
        )
    };

    const int w = src.width;
    const int h = src.height;
    const int ch = src.channels;

    const unsigned char* s = src.data.data();
    unsigned char* d = dst.data.data();

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < h; ++y) {
        const size_t row_offset = static_cast<size_t>(y) * w;
        const unsigned char* row_src = s + row_offset * ch;
        unsigned char* row_dst = d + row_offset;

        if (ch >= 3) {
            for (int x = 0; x < w; ++x) {
                const size_t p = static_cast<size_t>(x) * ch;

                float gray =
                    0.299f * row_src[p] +
                    0.587f * row_src[p + 1] +
                    0.114f * row_src[p + 2];

                row_dst[x] = clamp_u8(gray);
            }
        } else {
            for (int x = 0; x < w; ++x) {
                float v = row_src[static_cast<size_t>(x) * ch];
                row_dst[x] = clamp_u8(v);
            }
        }
    }

    return dst;
}

// GRAYSCALE: TILED + OPENMP
inline Image gray_tiled_openmp(const Image& src, int tile = 32) {
    Image dst{src.width, src.height, 1,
        std::vector<unsigned char>(
            static_cast<size_t>(src.width) * src.height
        )
    };

    const int w = src.width;
    const int h = src.height;
    const int ch = src.channels;

    const unsigned char* s = src.data.data();
    unsigned char* d = dst.data.data();

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int ty = 0; ty < h; ty += tile) {
        for (int tx = 0; tx < w; tx += tile) {
            int y_end = std::min(ty + tile, h);
            int x_end = std::min(tx + tile, w);

            for (int y = ty; y < y_end; ++y) {
                const size_t row_offset = static_cast<size_t>(y) * w;

                for (int x = tx; x < x_end; ++x) {
                    const size_t p = (row_offset + x) * ch;

                    float r = s[p];
                    float g = (ch > 1) ? s[p + 1] : r;
                    float b = (ch > 2) ? s[p + 2] : r;

                    d[row_offset + x] =
                        clamp_u8(0.299f * r + 0.587f * g + 0.114f * b);
                }
            }
        }
    }

    return dst;
}


// GAUSSIAN BLUR: BASELINE SERIAL
inline Image gauss_baseline_serial(const Image& src) {
    const float k[3][3] = {
        {1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f},
        {2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f},
        {1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f}
    };

    Image dst{src.width, src.height, src.channels,
              std::vector<unsigned char>(src.data.size())};

    int ch = src.channels;

    for (int y = 0; y < src.height; ++y) {
        for (int x = 0; x < src.width; ++x) {
            for (int c = 0; c < ch; ++c) {
                float sum = 0.0f;

                for (int ky = -1; ky <= 1; ++ky) {
                    int sy = std::clamp(y + ky, 0, src.height - 1);

                    for (int kx = -1; kx <= 1; ++kx) {
                        int sx = std::clamp(x + kx, 0, src.width - 1);

                        size_t idx =
                            (static_cast<size_t>(sy) * src.width + sx) * ch + c;

                        sum += src.data[idx] * k[ky + 1][kx + 1];
                    }
                }

                dst.data[(static_cast<size_t>(y) * src.width + x) * ch + c] =
                    clamp_u8(sum);
            }
        }
    }

    return dst;
}

// GAUSSIAN BLUR: CACHE-FRIENDLY
inline Image gauss_cache_friendly(const Image& src) {
    const float k[3][3] = {
        {1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f},
        {2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f},
        {1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f}
    };

    Image dst{src.width, src.height, src.channels,
              std::vector<unsigned char>(src.data.size())};

    const int w = src.width;
    const int h = src.height;
    const int ch = src.channels;

    const unsigned char* s = src.data.data();
    unsigned char* d = dst.data.data();

    for (int y = 0; y < h; ++y) {
        int y0 = std::max(0, y - 1);
        int y1 = y;
        int y2 = std::min(h - 1, y + 1);

        for (int x = 0; x < w; ++x) {
            int x0 = std::max(0, x - 1);
            int x1 = x;
            int x2 = std::min(w - 1, x + 1);

            const size_t r0 = static_cast<size_t>(y0) * w * ch;
            const size_t r1 = static_cast<size_t>(y1) * w * ch;
            const size_t r2 = static_cast<size_t>(y2) * w * ch;

            for (int c = 0; c < ch; ++c) {
                float sum =
                    s[r0 + x0 * ch + c] * k[0][0] +
                    s[r0 + x1 * ch + c] * k[0][1] +
                    s[r0 + x2 * ch + c] * k[0][2] +
                    s[r1 + x0 * ch + c] * k[1][0] +
                    s[r1 + x1 * ch + c] * k[1][1] +
                    s[r1 + x2 * ch + c] * k[1][2] +
                    s[r2 + x0 * ch + c] * k[2][0] +
                    s[r2 + x1 * ch + c] * k[2][1] +
                    s[r2 + x2 * ch + c] * k[2][2];

                d[(static_cast<size_t>(y) * w + x) * ch + c] =
                    clamp_u8(sum);
            }
        }
    }

    return dst;
}

// GAUSSIAN BLUR: OPENMP
inline Image gauss_openmp(const Image& src) {
    const float k[3][3] = {
        {1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f},
        {2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f},
        {1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f}
    };

    Image dst{src.width, src.height, src.channels,
              std::vector<unsigned char>(src.data.size())};

    const int w = src.width;
    const int h = src.height;
    const int ch = src.channels;

    const unsigned char* s = src.data.data();
    unsigned char* d = dst.data.data();

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < h; ++y) {
        int y0 = std::max(0, y - 1);
        int y1 = y;
        int y2 = std::min(h - 1, y + 1);

        for (int x = 0; x < w; ++x) {
            int x0 = std::max(0, x - 1);
            int x1 = x;
            int x2 = std::min(w - 1, x + 1);

            const size_t r0 = static_cast<size_t>(y0) * w * ch;
            const size_t r1 = static_cast<size_t>(y1) * w * ch;
            const size_t r2 = static_cast<size_t>(y2) * w * ch;

            for (int c = 0; c < ch; ++c) {
                float sum =
                    s[r0 + x0 * ch + c] * k[0][0] +
                    s[r0 + x1 * ch + c] * k[0][1] +
                    s[r0 + x2 * ch + c] * k[0][2] +
                    s[r1 + x0 * ch + c] * k[1][0] +
                    s[r1 + x1 * ch + c] * k[1][1] +
                    s[r1 + x2 * ch + c] * k[1][2] +
                    s[r2 + x0 * ch + c] * k[2][0] +
                    s[r2 + x1 * ch + c] * k[2][1] +
                    s[r2 + x2 * ch + c] * k[2][2];

                d[(static_cast<size_t>(y) * w + x) * ch + c] =
                    clamp_u8(sum);
            }
        }
    }

    return dst;
}

// GAUSSIAN BLUR: TILED + OPENMP
inline Image gauss_tiled_openmp(const Image& src, int tile = 32) {
    const float k[3][3] = {
        {1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f},
        {2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f},
        {1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f}
    };

    Image dst{src.width, src.height, src.channels,
              std::vector<unsigned char>(src.data.size())};

    const int w = src.width;
    const int h = src.height;
    const int ch = src.channels;

    const unsigned char* s = src.data.data();
    unsigned char* d = dst.data.data();

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int ty = 0; ty < h; ty += tile) {
        for (int tx = 0; tx < w; tx += tile) {
            int y_end = std::min(ty + tile, h);
            int x_end = std::min(tx + tile, w);

            for (int y = ty; y < y_end; ++y) {
                int y0 = std::max(0, y - 1);
                int y1 = y;
                int y2 = std::min(h - 1, y + 1);

                const size_t r0 = static_cast<size_t>(y0) * w * ch;
                const size_t r1 = static_cast<size_t>(y1) * w * ch;
                const size_t r2 = static_cast<size_t>(y2) * w * ch;

                for (int x = tx; x < x_end; ++x) {
                    int x0 = std::max(0, x - 1);
                    int x1 = x;
                    int x2 = std::min(w - 1, x + 1);

                    for (int c = 0; c < ch; ++c) {
                        float sum =
                            s[r0 + x0 * ch + c] * k[0][0] +
                            s[r0 + x1 * ch + c] * k[0][1] +
                            s[r0 + x2 * ch + c] * k[0][2] +
                            s[r1 + x0 * ch + c] * k[1][0] +
                            s[r1 + x1 * ch + c] * k[1][1] +
                            s[r1 + x2 * ch + c] * k[1][2] +
                            s[r2 + x0 * ch + c] * k[2][0] +
                            s[r2 + x1 * ch + c] * k[2][1] +
                            s[r2 + x2 * ch + c] * k[2][2];

                        d[(static_cast<size_t>(y) * w + x) * ch + c] =
                            clamp_u8(sum);
                    }
                }
            }
        }
    }

    return dst;
}


inline Image sobel_baseline_serial(const Image& src) {
    Image gray = (src.channels == 1) ? src : gray_baseline_serial(src);

    Image dst{gray.width, gray.height, 1,
        std::vector<unsigned char>(static_cast<size_t>(gray.width) * gray.height)};

    const int gx_k[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    const int gy_k[3][3] = {
        {-1, -2, -1},
        {0, 0, 0},
        {1, 2, 1}
    };

    for (int y = 0; y < gray.height; ++y) {
        for (int x = 0; x < gray.width; ++x) {
            float gx = 0.0f, gy = 0.0f;

            for (int ky = -1; ky <= 1; ++ky) {
                int sy = std::clamp(y + ky, 0, gray.height - 1);

                for (int kx = -1; kx <= 1; ++kx) {
                    int sx = std::clamp(x + kx, 0, gray.width - 1);

                    float p = static_cast<float>(
                        gray.data[static_cast<size_t>(sy) * gray.width + sx]
                    );

                    gx += p * gx_k[ky + 1][kx + 1];
                    gy += p * gy_k[ky + 1][kx + 1];
                }
            }

            dst.data[static_cast<size_t>(y) * gray.width + x] =
                clamp_u8(std::sqrt(gx * gx + gy * gy));
        }
    }

    return dst;
}

inline Image sobel_cache_friendly(const Image& src) {
    Image gray = (src.channels == 1) ? src : gray_cache_friendly(src);

    const int w = gray.width;
    const int h = gray.height;

    Image dst{w, h, 1,
        std::vector<unsigned char>(static_cast<size_t>(w) * h)};

    const unsigned char* g = gray.data.data();
    unsigned char* d = dst.data.data();

    for (int y = 0; y < h; ++y) {
        int y0 = std::max(0, y - 1);
        int y1 = y;
        int y2 = std::min(h - 1, y + 1);

        const unsigned char* r0 = g + static_cast<size_t>(y0) * w;
        const unsigned char* r1 = g + static_cast<size_t>(y1) * w;
        const unsigned char* r2 = g + static_cast<size_t>(y2) * w;

        unsigned char* out_row = d + static_cast<size_t>(y) * w;

        for (int x = 0; x < w; ++x) {
            int x0 = std::max(0, x - 1);
            int x1 = x;
            int x2 = std::min(w - 1, x + 1);

            float p00 = r0[x0], p01 = r0[x1], p02 = r0[x2];
            float p10 = r1[x0], p12 = r1[x2];
            float p20 = r2[x0], p21 = r2[x1], p22 = r2[x2];

            float gx = (-p00 + p02) + (-2.0f * p10 + 2.0f * p12) + (-p20 + p22);
            float gy = (-p00 - 2.0f * p01 - p02) + (p20 + 2.0f * p21 + p22);

            out_row[x] = clamp_u8(std::sqrt(gx * gx + gy * gy));
        }
    }

    return dst;
}

inline Image sobel_openmp(const Image& src) {
    Image gray = (src.channels == 1) ? src : gray_openmp(src);

    const int w = gray.width;
    const int h = gray.height;

    Image dst{w, h, 1,
        std::vector<unsigned char>(static_cast<size_t>(w) * h)};

    const unsigned char* g = gray.data.data();
    unsigned char* d = dst.data.data();

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < h; ++y) {
        int y0 = std::max(0, y - 1);
        int y1 = y;
        int y2 = std::min(h - 1, y + 1);

        const unsigned char* r0 = g + static_cast<size_t>(y0) * w;
        const unsigned char* r1 = g + static_cast<size_t>(y1) * w;
        const unsigned char* r2 = g + static_cast<size_t>(y2) * w;

        unsigned char* out_row = d + static_cast<size_t>(y) * w;

        for (int x = 0; x < w; ++x) {
            int x0 = std::max(0, x - 1);
            int x1 = x;
            int x2 = std::min(w - 1, x + 1);

            float p00 = r0[x0], p01 = r0[x1], p02 = r0[x2];
            float p10 = r1[x0], p12 = r1[x2];
            float p20 = r2[x0], p21 = r2[x1], p22 = r2[x2];

            float gx = (-p00 + p02) + (-2.0f * p10 + 2.0f * p12) + (-p20 + p22);
            float gy = (-p00 - 2.0f * p01 - p02) + (p20 + 2.0f * p21 + p22);

            out_row[x] = clamp_u8(std::sqrt(gx * gx + gy * gy));
        }
    }

    return dst;
}

inline Image sobel_tiled_openmp(const Image& src, int tile = 32) {
    Image gray = (src.channels == 1) ? src : gray_tiled_openmp(src, tile);

    const int w = gray.width;
    const int h = gray.height;

    Image dst{w, h, 1,
        std::vector<unsigned char>(static_cast<size_t>(w) * h)};

    const unsigned char* g = gray.data.data();
    unsigned char* d = dst.data.data();

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int ty = 0; ty < h; ty += tile) {
        for (int tx = 0; tx < w; tx += tile) {
            int y_end = std::min(ty + tile, h);
            int x_end = std::min(tx + tile, w);

            for (int y = ty; y < y_end; ++y) {
                int y0 = std::max(0, y - 1);
                int y1 = y;
                int y2 = std::min(h - 1, y + 1);

                const unsigned char* r0 = g + static_cast<size_t>(y0) * w;
                const unsigned char* r1 = g + static_cast<size_t>(y1) * w;
                const unsigned char* r2 = g + static_cast<size_t>(y2) * w;

                unsigned char* out_row = d + static_cast<size_t>(y) * w;

                for (int x = tx; x < x_end; ++x) {
                    int x0 = std::max(0, x - 1);
                    int x1 = x;
                    int x2 = std::min(w - 1, x + 1);

                    float p00 = r0[x0], p01 = r0[x1], p02 = r0[x2];
                    float p10 = r1[x0], p12 = r1[x2];
                    float p20 = r2[x0], p21 = r2[x1], p22 = r2[x2];

                    float gx = (-p00 + p02) + (-2.0f * p10 + 2.0f * p12) + (-p20 + p22);
                    float gy = (-p00 - 2.0f * p01 - p02) + (p20 + 2.0f * p21 + p22);

                    out_row[x] = clamp_u8(std::sqrt(gx * gx + gy * gy));
                }
            }
        }
    }

    return dst;
}

#endif

