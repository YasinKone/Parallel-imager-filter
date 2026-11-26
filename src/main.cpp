#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

struct Image {
    int width = 0;
    int height = 0;
    int channels = 0;
    std::vector<unsigned char> data;
};

struct StrategyResult {
    std::string name;
    std::string filename;
    double ms = 0.0;
    int max_diff_vs_baseline = 0;
    Image image;
};

static inline unsigned char clamp_u8(float val) {
    if (val < 0.0f) return 0;
    if (val > 255.0f) return 255;
    return static_cast<unsigned char>(std::lround(val));
}

bool load_image(const std::string& path, Image& img) {
    int w, h, c;
    unsigned char* raw = stbi_load(path.c_str(), &w, &h, &c, 0);
    if (!raw) {
        std::cerr << "Failed to load image: " << path << "\n";
        return false;
    }
    img.width = w;
    img.height = h;
    img.channels = c;
    img.data.assign(raw, raw + static_cast<size_t>(w) * h * c);
    stbi_image_free(raw);
    return true;
}

bool save_image(const std::string& path, const Image& img) {
    if (stbi_write_png(path.c_str(), img.width, img.height, img.channels,
                       img.data.data(), img.width * img.channels) == 0) {
        std::cerr << "Failed to write output image: " << path << "\n";
        return false;
    }
    return true;
}

int compute_max_diff(const Image& a, const Image& b) {
    if (a.width != b.width || a.height != b.height || a.channels != b.channels ||
        a.data.size() != b.data.size()) {
        return -1;
    }
    int max_d = 0;
    for (size_t i = 0; i < a.data.size(); ++i) {
        int d = std::abs(static_cast<int>(a.data[i]) - static_cast<int>(b.data[i]));
        if (d > max_d) max_d = d;
    }
    return max_d;
}

template <typename Func>
double measure_ms(Func&& fn) {
    auto start = std::chrono::high_resolution_clock::now();
    fn();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}


// GRAYSCALE STRATEGIES


Image gray_baseline_serial(const Image& src) {
    Image dst{src.width, src.height, 1,
              std::vector<unsigned char>(static_cast<size_t>(src.width) * src.height)};
    int ch = src.channels;
    for (int y = 0; y < src.height; ++y) {
        for (int x = 0; x < src.width; ++x) {
            size_t idx = (static_cast<size_t>(y) * src.width + x) * ch;
            float r = src.data[idx + 0];
            float g = (ch > 1) ? src.data[idx + 1] : r;
            float b = (ch > 2) ? src.data[idx + 2] : r;
            float gray = 0.299f * r + 0.587f * g + 0.114f * b;
            dst.data[static_cast<size_t>(y) * src.width + x] = clamp_u8(gray);
        }
    }
    return dst;
}

Image gray_cache_friendly(const Image& src) {
    Image dst{src.width, src.height, 1,
              std::vector<unsigned char>(static_cast<size_t>(src.width) * src.height)};
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
                float gray = 0.299f * row_src[p] + 0.587f * row_src[p + 1] + 0.114f * row_src[p + 2];
                row_dst[x] = clamp_u8(gray);
            }
        } else if (ch == 2) {
            for (int x = 0; x < w; ++x) {
                const size_t p = static_cast<size_t>(x) * 2;
                float r = row_src[p];
                float g = row_src[p + 1];
                row_dst[x] = clamp_u8(0.299f * r + 0.587f * g + 0.114f * r);
            }
        } else {
            std::copy(row_src, row_src + w, row_dst);
        }
    }
    return dst;
}

Image gray_openmp(const Image& src) {
    Image dst{src.width, src.height, 1,
              std::vector<unsigned char>(static_cast<size_t>(src.width) * src.height)};
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
                float gray = 0.299f * row_src[p] + 0.587f * row_src[p + 1] + 0.114f * row_src[p + 2];
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

Image gray_tiled_openmp(const Image& src, int tile = 32) {
    Image dst{src.width, src.height, 1,
              std::vector<unsigned char>(static_cast<size_t>(src.width) * src.height)};
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
                    d[row_offset + x] = clamp_u8(0.299f * r + 0.587f * g + 0.114f * b);
                }
            }
        }
    }
    return dst;
}


// GAUSSIAN BLUR STRATEGIES


Image gauss_baseline_serial(const Image& src) {
    const float k[3][3] = {
        {1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f},
        {2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f},
        {1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f}
    };
    Image dst{src.width, src.height, src.channels, std::vector<unsigned char>(src.data.size())};
    int ch = src.channels;

    for (int y = 0; y < src.height; ++y) {
        for (int x = 0; x < src.width; ++x) {
            for (int c = 0; c < ch; ++c) {
                float sum = 0.0f;
                for (int ky = -1; ky <= 1; ++ky) {
                    int sy = std::clamp(y + ky, 0, src.height - 1);
                    for (int kx = -1; kx <= 1; ++kx) {
                        int sx = std::clamp(x + kx, 0, src.width - 1);
                        size_t idx = (static_cast<size_t>(sy) * src.width + sx) * ch + c;
                        sum += src.data[idx] * k[ky + 1][kx + 1];
                    }
                }
                dst.data[(static_cast<size_t>(y) * src.width + x) * ch + c] = clamp_u8(sum);
            }
        }
    }
    return dst;
}

Image gauss_cache_friendly(const Image& src) {
    const float k[3][3] = {
        {1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f},
        {2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f},
        {1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f}
    };
    Image dst{src.width, src.height, src.channels, std::vector<unsigned char>(src.data.size())};
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
                    s[r0 + x0 * ch + c] * k[0][0] + s[r0 + x1 * ch + c] * k[0][1] + s[r0 + x2 * ch + c] * k[0][2] +
                    s[r1 + x0 * ch + c] * k[1][0] + s[r1 + x1 * ch + c] * k[1][1] + s[r1 + x2 * ch + c] * k[1][2] +
                    s[r2 + x0 * ch + c] * k[2][0] + s[r2 + x1 * ch + c] * k[2][1] + s[r2 + x2 * ch + c] * k[2][2];
                d[(static_cast<size_t>(y) * w + x) * ch + c] = clamp_u8(sum);
            }
        }
    }
    return dst;
}

Image gauss_openmp(const Image& src) {
    const float k[3][3] = {
        {1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f},
        {2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f},
        {1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f}
    };
    Image dst{src.width, src.height, src.channels, std::vector<unsigned char>(src.data.size())};
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
                    s[r0 + x0 * ch + c] * k[0][0] + s[r0 + x1 * ch + c] * k[0][1] + s[r0 + x2 * ch + c] * k[0][2] +
                    s[r1 + x0 * ch + c] * k[1][0] + s[r1 + x1 * ch + c] * k[1][1] + s[r1 + x2 * ch + c] * k[1][2] +
                    s[r2 + x0 * ch + c] * k[2][0] + s[r2 + x1 * ch + c] * k[2][1] + s[r2 + x2 * ch + c] * k[2][2];
                d[(static_cast<size_t>(y) * w + x) * ch + c] = clamp_u8(sum);
            }
        }
    }
    return dst;
}

Image gauss_tiled_openmp(const Image& src, int tile = 32) {
    const float k[3][3] = {
        {1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f},
        {2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f},
        {1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f}
    };
    Image dst{src.width, src.height, src.channels, std::vector<unsigned char>(src.data.size())};
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
                            s[r0 + x0 * ch + c] * k[0][0] + s[r0 + x1 * ch + c] * k[0][1] + s[r0 + x2 * ch + c] * k[0][2] +
                            s[r1 + x0 * ch + c] * k[1][0] + s[r1 + x1 * ch + c] * k[1][1] + s[r1 + x2 * ch + c] * k[1][2] +
                            s[r2 + x0 * ch + c] * k[2][0] + s[r2 + x1 * ch + c] * k[2][1] + s[r2 + x2 * ch + c] * k[2][2];
                        d[(static_cast<size_t>(y) * w + x) * ch + c] = clamp_u8(sum);
                    }
                }
            }
        }
    }
    return dst;
}


// SOBEL EDGE DETECTION STRATEGIES


Image sobel_baseline_serial(const Image& src) {
    Image gray = (src.channels == 1) ? src : gray_baseline_serial(src);
    Image dst{gray.width, gray.height, 1,
              std::vector<unsigned char>(static_cast<size_t>(gray.width) * gray.height)};
    const int gx_k[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    const int gy_k[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    for (int y = 0; y < gray.height; ++y) {
        for (int x = 0; x < gray.width; ++x) {
            float gx = 0.0f, gy = 0.0f;
            for (int ky = -1; ky <= 1; ++ky) {
                int sy = std::clamp(y + ky, 0, gray.height - 1);
                for (int kx = -1; kx <= 1; ++kx) {
                    int sx = std::clamp(x + kx, 0, gray.width - 1);
                    float p = static_cast<float>(gray.data[static_cast<size_t>(sy) * gray.width + sx]);
                    gx += p * gx_k[ky + 1][kx + 1];
                    gy += p * gy_k[ky + 1][kx + 1];
                }
            }
            dst.data[static_cast<size_t>(y) * gray.width + x] = clamp_u8(std::sqrt(gx * gx + gy * gy));
        }
    }
    return dst;
}

Image sobel_cache_friendly(const Image& src) {
    Image gray = (src.channels == 1) ? src : gray_cache_friendly(src);
    const int w = gray.width;
    const int h = gray.height;
    Image dst{w, h, 1, std::vector<unsigned char>(static_cast<size_t>(w) * h)};
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
            float p10 = r1[x0],               p12 = r1[x2];
            float p20 = r2[x0], p21 = r2[x1], p22 = r2[x2];

            float gx = (-p00 + p02) + (-2.0f * p10 + 2.0f * p12) + (-p20 + p22);
            float gy = (-p00 - 2.0f * p01 - p02) + (p20 + 2.0f * p21 + p22);
            out_row[x] = clamp_u8(std::sqrt(gx * gx + gy * gy));
        }
    }
    return dst;
}

Image sobel_openmp(const Image& src) {
    Image gray = (src.channels == 1) ? src : gray_openmp(src);
    const int w = gray.width;
    const int h = gray.height;
    Image dst{w, h, 1, std::vector<unsigned char>(static_cast<size_t>(w) * h)};
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
            float p10 = r1[x0],               p12 = r1[x2];
            float p20 = r2[x0], p21 = r2[x1], p22 = r2[x2];

            float gx = (-p00 + p02) + (-2.0f * p10 + 2.0f * p12) + (-p20 + p22);
            float gy = (-p00 - 2.0f * p01 - p02) + (p20 + 2.0f * p21 + p22);
            out_row[x] = clamp_u8(std::sqrt(gx * gx + gy * gy));
        }
    }
    return dst;
}

Image sobel_tiled_openmp(const Image& src, int tile = 32) {
    Image gray = (src.channels == 1) ? src : gray_tiled_openmp(src, tile);
    const int w = gray.width;
    const int h = gray.height;
    Image dst{w, h, 1, std::vector<unsigned char>(static_cast<size_t>(w) * h)};
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
                    float p10 = r1[x0],               p12 = r1[x2];
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


// BENCHMARK & COMPARISON RUNNER


void print_comparison_table(const std::string& filter_label, const std::vector<StrategyResult>& results) {
    double base_ms = results.empty() ? 1.0 : results[0].ms;
    std::cout << "\n=== Comparison: " << filter_label << " ===\n";
    std::cout << std::left
              << std::setw(28) << "Strategy"
              << std::right
              << std::setw(12) << "Time (ms)"
              << std::setw(12) << "Speedup"
              << std::setw(12) << "Max Diff"
              << "  " << std::left << "Output File\n";
    std::cout << std::string(86, '-') << "\n";

    for (const auto& r : results) {
        double speedup = (r.ms > 0.0) ? (base_ms / r.ms) : 0.0;
        std::cout << std::left
                  << std::setw(28) << r.name
                  << std::right
                  << std::setw(12) << std::fixed << std::setprecision(3) << r.ms
                  << std::setw(11) << std::setprecision(2) << speedup << "x"
                  << std::setw(12) << r.max_diff_vs_baseline
                  << "  " << std::left << r.filename << "\n";
    }
}

void run_filter_all_strategies(int choice, const Image& src) {
    std::vector<StrategyResult> results;

    auto record = [&](const std::string& name, const std::string& file, auto&& fn) {
        StrategyResult r;
        r.name = name;
        r.filename = file;
        r.ms = measure_ms([&]() { r.image = fn(); });
        r.max_diff_vs_baseline = results.empty() ? 0 : compute_max_diff(results[0].image, r.image);
        save_image(r.filename, r.image);
        results.push_back(std::move(r));
    };

    if (choice == 1) {
        record("1. Baseline Serial", "out_gray_1_serial.png", [&]() { return gray_baseline_serial(src); });
        record("2. Cache-Friendly", "out_gray_2_cache.png", [&]() { return gray_cache_friendly(src); });
        record("4. OpenMP Parallel", "out_gray_4_openmp.png", [&]() { return gray_openmp(src); });
        record("5. Tiled + OpenMP", "out_gray_5_tiled.png", [&]() { return gray_tiled_openmp(src, 32); });
        print_comparison_table("Grayscale Conversion", results);
    } else if (choice == 2) {
        record("1. Baseline Serial", "out_gauss_1_serial.png", [&]() { return gauss_baseline_serial(src); });
        record("2. Cache-Friendly", "out_gauss_2_cache.png", [&]() { return gauss_cache_friendly(src); });
        record("4. OpenMP Parallel", "out_gauss_4_openmp.png", [&]() { return gauss_openmp(src); });
        record("5. Tiled + OpenMP", "out_gauss_5_tiled.png", [&]() { return gauss_tiled_openmp(src, 32); });
        print_comparison_table("Gaussian Blur", results);
    } else if (choice == 3) {
        record("1. Baseline Serial", "out_sobel_1_serial.png", [&]() { return sobel_baseline_serial(src); });
        record("2. Cache-Friendly", "out_sobel_2_cache.png", [&]() { return sobel_cache_friendly(src); });
        record("4. OpenMP Parallel", "out_sobel_4_openmp.png", [&]() { return sobel_openmp(src); });
        record("5. Tiled + OpenMP", "out_sobel_5_tiled.png", [&]() { return sobel_tiled_openmp(src, 32); });
        print_comparison_table("Sobel Edge Detection", results);
    }
}

void print_menu(const std::string& filename) {
    int threads = 1;
    #ifdef _OPENMP
    #pragma omp parallel
    {
        #pragma omp single
        threads = omp_get_num_threads();
    }
    #endif
    std::cout << "\n--- Image Processing Benchmark CLI (OpenMP threads: " << threads << ") ---\n";
    std::cout << "Target file: " << filename << "\n";
    std::cout << "1. Grayscale Conversion\n";
    std::cout << "2. Gaussian Blur\n";
    std::cout << "3. Sobel Edge Detection)\n";
    std::cout << "4. Change Image Filename / Exit\n";
    std::cout << "Enter option (1-4): ";
}

int main() {
    std::string filename;
    std::cout << "Enter image filename in project folder (e.g., sample.png): ";
    std::getline(std::cin, filename);

    Image img;
    while (!load_image(filename, img)) {
        std::cout << "Try another filename (or type 'exit' to quit): ";
        std::getline(std::cin, filename);
        if (filename == "exit") return 0;
    }

    std::cout << "Loaded " << filename << " (" << img.width << "x" << img.height
              << ", channels=" << img.channels << ")\n";

    while (true) {
        print_menu(filename);
        int choice = 0;
        if (!(std::cin >> choice)) {
            std::cin.clear();
            std::cin.ignore(10000, '\n');
            std::cout << "Invalid input. Please enter 1, 2, 3, or 4.\n";
            continue;
        }
        std::cin.ignore(10000, '\n');

        if (choice == 4) {
            std::cout << "Enter new filename (or 'exit' to quit): ";
            std::getline(std::cin, filename);
            if (filename == "exit") break;
            Image next_img;
            if (load_image(filename, next_img)) {
                img = std::move(next_img);
                std::cout << "Loaded " << filename << " (" << img.width << "x" << img.height
                          << ", channels=" << img.channels << ")\n";
            }
            continue;
        }

        if (choice >= 1 && choice <= 3) {
            run_filter_all_strategies(choice, img);
        } else {
            std::cout << "Please select a valid option (1-4).\n";
        }
    }

    return 0;
}