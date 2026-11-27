
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "implot.h"
#include "../include/image_filters.h"

#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h"

#include <GLFW/glfw3.h>
#include <iostream>
#include <string>
#include <chrono>
#include <cstdio>

#ifdef _OPENMP
#include <omp.h>
#endif 

std::string open_image_file_picker();



Image resize_image_nearest(const Image& source, int new_width, int new_height) {
    Image resized;

    resized.width = new_width;
    resized.height = new_height;
    resized.channels = source.channels;

    resized.data.resize(
        static_cast<size_t>(new_width) * new_height * source.channels
    );

    for (int y = 0; y < new_height; y++) {
        for (int x = 0; x < new_width; x++) {

            int source_x = x * source.width / new_width;
            int source_y = y * source.height / new_height;

            for (int c = 0; c < source.channels; c++) {

                size_t source_index =
                    (static_cast<size_t>(source_y) * source.width + source_x)
                    * source.channels + c;

                size_t destination_index =
                    (static_cast<size_t>(y) * new_width + x)
                    * source.channels + c;

                resized.data[destination_index] =
                    source.data[source_index];
            }
        }
    }

    return resized;
}


Image apply_filter(const Image& source, int filter, int implementation) {

    switch (filter) {

        case 0:
            switch (implementation) {
                case 0: return gray_baseline_serial(source);
                case 1: return gray_cache_friendly(source);
                case 2: return gray_openmp(source);
                case 3: return gray_tiled_openmp(source);
            }
            break;

        case 1:
            switch (implementation) {
                case 0: return gauss_baseline_serial(source);
                case 1: return gauss_cache_friendly(source);
                case 2: return gauss_openmp(source);
                case 3: return gauss_tiled_openmp(source);
            }
            break;

        case 2:
            switch (implementation) {
                case 0: return sobel_baseline_serial(source);
                case 1: return sobel_cache_friendly(source);
                case 2: return sobel_openmp(source);
                case 3: return sobel_tiled_openmp(source);
            }
            break;
    }

    return source;
}


int main() {

    if (!glfwInit()) {
        std::cout << "Failed to initialize GLFW\n";
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    GLFWwindow* window = glfwCreateWindow(
        1400, 900,
        "Parallel Image Filtering",
        nullptr, nullptr
    );

    if (!window) {
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 150");

    // Application selections
    static int selected_filter = 0;
    static int selected_implementation = 0;
    static int selected_threads = 4;

    unsigned char* original_pixels = nullptr;

    int original_width = 0;
    int original_height = 0;
    int original_channels = 0;

    GLuint original_texture = 0;

    std::string image_path ="";
    bool image_loaded = false;

    GLuint processed_texture = 0;

    Image processed_image;
    bool processed_image_loaded = false;

    double execution_time_ms = 0.0;
    double serial_baseline_ms[3] = {0.0, 0.0, 0.0};

    double benchmark_times_ms[3][4] = {};
    double thread_benchmark_times_ms[3][4][8] = {};

    // Image resolution benchmark data
    double resolution_widths[4] = {};
    double resolution_times_ms[4] = {};
    int resolution_measurement_count = 0;

    const char* filters[] = {
        "Grayscale",
        "Gaussian Blur",
        "Sobel Edge Detection"
    };

    const char* implementations[] = {
        "Serial",
        "Cache-Friendly",
        "OpenMP",
        "Tiled + OpenMP"
    };

    // Sample graph data - placeholder only
    double thread_counts[] = {1, 2, 4, 8};
    double speedup[] = {1.0, 1.94, 3.75, 5.18};

    double versions[] = {1, 2, 3, 4};
    double execution_times[] = {33.3, 9.04, 3.15, 2.47};

    double resolutions[] = {600, 1200, 1800, 2400};
    double resolution_times[] = {5.2, 14.3, 29.8, 51.2};

    double efficiency[] = {100, 97, 94, 65};

    while (!glfwWindowShouldClose(window)) {

        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);

        ImGui::Begin(
            "Parallel Image Filtering",
            nullptr,
            ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoCollapse
        );

        ImGui::Text("PARALLEL IMAGE FILTERING");
        ImGui::SameLine();
        ImGui::TextDisabled("| Performance and Scalability");
        ImGui::Separator();

        if (ImGui::BeginTabBar("MainTabs")) {

            // FILTERS TAB
            if (ImGui::BeginTabItem("Filters")) {

                ImGui::Spacing();

                // Left sidebar
                ImGui::BeginChild(
                    "Sidebar",
                    ImVec2(270, 0),
                    true
                );

                ImGui::Text("IMAGE PROCESSING");
                ImGui::Separator();

                ImGui::Text("Select Filter");

                ImGui::Combo(
                    "##Filter",
                    &selected_filter,
                    filters,
                    IM_ARRAYSIZE(filters)
                );

                ImGui::Spacing();
                ImGui::Text("Implementation");

                ImGui::Combo(
                    "##Implementation",
                    &selected_implementation,
                    implementations,
                    IM_ARRAYSIZE(implementations)
                );

                ImGui::Spacing();
                ImGui::Text("Thread Count");

                ImGui::SliderInt(
                    "##Threads",
                    &selected_threads,
                    1,
                    8
                );

                ImGui::Spacing();
                ImGui::Separator();

                if (ImGui::Button("Load Image", ImVec2(-1, 40))) {

    std::string selected_path = open_image_file_picker();

    if (!selected_path.empty()) {

        if (original_pixels != nullptr) {
            stbi_image_free(original_pixels);
            original_pixels = nullptr;
        }

        if (original_texture != 0) {
            glDeleteTextures(1, &original_texture);
            original_texture = 0;
        }

        original_pixels = stbi_load(
            selected_path.c_str(),
            &original_width,
            &original_height,
            &original_channels,
            4
        );

        if (original_pixels != nullptr) {
            image_path = selected_path;
            image_loaded = true;

            glGenTextures(1, &original_texture);
            glBindTexture(GL_TEXTURE_2D, original_texture);


            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); 

            glTexImage2D(
                GL_TEXTURE_2D,
                0,
                GL_RGBA,
                original_width,
                original_height,
                0,
                GL_RGBA,
                GL_UNSIGNED_BYTE,
                original_pixels
            );

            glBindTexture(GL_TEXTURE_2D, 0);


            std::cout << "Image loaded successfully!\n";
            std::cout << "Dimensions: "
                      << original_width << "x"
                      << original_height << "\n";
        } else {
            image_loaded = false;
            std::cout << "Failed to load image.\n";
        }
    }
}


if (ImGui::Button("Run Filter", ImVec2(-1, 40))) {

    if (image_loaded && original_pixels != nullptr) {

        // Convert the loaded image into our Image structure
        Image source;

        source.width = original_width;
        source.height = original_height;
        source.channels = 4;

        source.data.assign(
            original_pixels,
            original_pixels +
            static_cast<size_t>(original_width) * original_height * 4
        );

        // Benchmark different image resolutions
resolution_measurement_count = 0;

double scale_factors[] = {0.25, 0.50, 0.75, 1.00};

#ifdef _OPENMP
omp_set_num_threads(selected_threads);
#endif

for (double scale_factor : scale_factors) {

    int test_width = std::max(
        1,
        static_cast<int>(source.width * scale_factor)
    );

    int test_height = std::max(
        1,
        static_cast<int>(source.height * scale_factor)
    );

    Image test_image = resize_image_nearest(
        source,
        test_width,
        test_height
    );

    // Warm-up
    Image result = apply_filter(
        test_image,
        selected_filter,
        selected_implementation
    );

    double total_time = 0.0;

    // Three measured runs
    for (int run = 0; run < 3; run++) {

        auto start = std::chrono::high_resolution_clock::now();

        result = apply_filter(
            test_image,
            selected_filter,
            selected_implementation
        );

        auto end = std::chrono::high_resolution_clock::now();

        total_time +=
            std::chrono::duration<double, std::milli>(
                end - start
            ).count();
    }

    resolution_widths[resolution_measurement_count] = test_width;

    resolution_times_ms[resolution_measurement_count] =
        total_time / 3.0;

    resolution_measurement_count++;
}


        // Apply the selected thread count
        #ifdef _OPENMP
        omp_set_num_threads(selected_threads);
        #endif

        // Start measuring execution time
        auto start = std::chrono::high_resolution_clock::now();

        // Select the filter and implementation
        processed_image = apply_filter(
    source,
    selected_filter,
    selected_implementation
);

        auto end = std::chrono::high_resolution_clock::now();

        execution_time_ms =
            std::chrono::duration<double, std::milli>(
                end - start
            ).count();

        if (selected_implementation == 0) {
                serial_baseline_ms[selected_filter] = execution_time_ms;
            }


            benchmark_times_ms[selected_filter][selected_implementation] = 
            execution_time_ms;



            // save execution time for OpenMP thread count
            if (selected_implementation >=2){

                thread_benchmark_times_ms
                [selected_filter]
                [selected_implementation]
                [selected_threads-1] = execution_time_ms;


         }





        processed_image_loaded = true;

        // Remove the previous processed texture
        if (processed_texture != 0) {
            glDeleteTextures(1, &processed_texture);
            processed_texture = 0;
        }

        // Convert processed image into RGBA for OpenGL display
        size_t pixel_count =
            static_cast<size_t>(processed_image.width) *
            processed_image.height;

        std::vector<unsigned char> processed_rgba(pixel_count * 4);

        for (size_t i = 0; i < pixel_count; ++i) {

            if (processed_image.channels == 1) {

                unsigned char gray = processed_image.data[i];

                processed_rgba[i * 4 + 0] = gray;
                processed_rgba[i * 4 + 1] = gray;
                processed_rgba[i * 4 + 2] = gray;
                processed_rgba[i * 4 + 3] = 255;

            } else {

                size_t source_index = i * processed_image.channels;

                processed_rgba[i * 4 + 0] =
                    processed_image.data[source_index];

                processed_rgba[i * 4 + 1] =
                    processed_image.data[source_index + 1];

                processed_rgba[i * 4 + 2] =
                    processed_image.data[source_index + 2];

                processed_rgba[i * 4 + 3] =
                    (processed_image.channels >= 4)
                    ? processed_image.data[source_index + 3]
                    : 255;
            }
        }

        // Create the OpenGL texture
        glGenTextures(1, &processed_texture);
        glBindTexture(GL_TEXTURE_2D, processed_texture);

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGBA,
            processed_image.width,
            processed_image.height,
            0,
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            processed_rgba.data()
        );

        glBindTexture(GL_TEXTURE_2D, 0);

        std::cout << filters[selected_filter] << " filter completed!\n";
        std::cout << "Implementation: "
                  << implementations[selected_implementation] << "\n";
        std::cout << "Execution time: "
                  << execution_time_ms << " ms\n";

    } else {
        std::cout << "Please load an image first.\n";
    }
}


    ImGui::Spacing();
                
if (image_loaded) {

// Extract just the filename from the full image path
    size_t last_slash = image_path.find_last_of("/\\");
    std::string filename = image_path.substr(last_slash + 1);

    ImGui::Text("Image: %s", filename.c_str());

    ImGui::Text(
        "Dimensions: %d x %d",
        original_width,
        original_height
    );

    ImGui::TextColored(
        ImVec4(0.3f, 0.8f, 0.4f, 1.0f),
        "Status: Ready"
    );

} else {

    ImGui::TextDisabled("Image: Not loaded");
    ImGui::TextDisabled("Status: Waiting");

}

                ImGui::EndChild();

                ImGui::SameLine();

                // Image preview area
                ImGui::BeginChild("PreviewArea", ImVec2(0, 0), true);

                ImGui::Text("IMAGE PREVIEW");
                ImGui::Separator();

                float preview_width = ImGui::GetContentRegionAvail().x;
                float panel_width = (preview_width - 20) / 2;

                ImGui::BeginChild(
                    "OriginalImage",
                    ImVec2(panel_width, 420),
                    true
                );

                ImGui::Text("Original Image");
                ImGui::Separator();

    if (image_loaded && original_texture != 0) {

    float available_width = ImGui::GetContentRegionAvail().x;
    float available_height = ImGui::GetContentRegionAvail().y;

    float scale_x = available_width / original_width;
    float scale_y = available_height / original_height;

    float scale = (scale_x < scale_y) ? scale_x : scale_y;

    float image_display_width = original_width * scale;
    float image_display_height = original_height * scale;

    ImGui::SetCursorPosX(
        ImGui::GetCursorPosX() + 
        (available_width - image_display_width) / 2
    );


    ImGui::Image(
        (ImTextureID)(intptr_t)original_texture,
        ImVec2(image_display_width, image_display_height)
    );


} else {
    ImGui::SetCursorPos(ImVec2(30, 180));
    ImGui::TextDisabled("No image loaded");
}



                ImGui::EndChild();

                ImGui::SameLine();

                ImGui::BeginChild(
                    "ProcessedImage",
                    ImVec2(0, 420),
                    true
                );
            //MARKER STARTS
 ImGui::Text("Processed Image");
ImGui::Separator();

if (processed_image_loaded && processed_texture != 0) {

    float available_width = ImGui::GetContentRegionAvail().x;
    float available_height = ImGui::GetContentRegionAvail().y;

    float scale_x = available_width / processed_image.width;
    float scale_y = available_height / processed_image.height;

    float scale = (scale_x < scale_y) ? scale_x : scale_y;

    float image_display_width = processed_image.width * scale;
    float image_display_height = processed_image.height * scale;

    ImGui::SetCursorPosX(
        ImGui::GetCursorPosX() +
        (available_width - image_display_width) / 2
    );

    ImGui::Image(
        (ImTextureID)(intptr_t)processed_texture,
        ImVec2(image_display_width, image_display_height)
    );

} else {

    ImGui::SetCursorPos(ImVec2(30, 180));
    ImGui::TextDisabled("No processed image");

}

        
                ImGui::EndChild();

                ImGui::Spacing();
                ImGui::Separator();

ImGui::Text("Execution Summary");
ImGui::Separator();

if (processed_image_loaded) {
    ImGui::Text("Execution Time: %.3f ms", execution_time_ms);

    if (serial_baseline_ms[selected_filter] > 0.0) {
        double speedup =
            serial_baseline_ms[selected_filter] / execution_time_ms;

        ImGui::Text("Speedup: %.2fx", speedup);
    } else {
        ImGui::TextDisabled("Speedup: Run Serial first");
    }

    ImGui::Text("Threads: %d", selected_threads);
} else {
    ImGui::TextDisabled("Execution time: --");
    ImGui::TextDisabled("Speedup: --");
    ImGui::TextDisabled("Threads: --");
}


                ImGui::EndChild();

                ImGui::EndTabItem();
            }

            // PERFORMANCE TAB
            if (ImGui::BeginTabItem("Performance")) {

                ImGui::Spacing();



                ImGui::Text("PERFORMANCE DASHBOARD");

                ImGui::Spacing();





                if (ImGui::Button("Run All Benchmarks", ImVec2(220, 40))) {

    if (image_loaded && original_pixels != nullptr) {

        // Convert the original image into our Image structure
        Image source;

        source.width = original_width;
        source.height = original_height;
        source.channels = 4;

        source.data.assign(
            original_pixels,
            original_pixels +
            static_cast<size_t>(original_width) * original_height * 4
        );

        // Clear previous benchmark results
        for (int f = 0; f < 3; f++) {

            serial_baseline_ms[f] = 0.0;

            for (int impl = 0; impl < 4; impl++) {

                benchmark_times_ms[f][impl] = 0.0;

                for (int t = 0; t < 8; t++) {
                    thread_benchmark_times_ms[f][impl][t] = 0.0;
                }
            }
        }

        // Test every filter
        for (int f = 0; f < 3; f++) {

            // Test every implementation
            for (int impl = 0; impl < 4; impl++) {

                // Serial and Cache-Friendly
                if (impl < 2) {

                    Image result;

                    // Warm-up run
                    result = apply_filter(source, f, impl);

                    double total_time = 0.0;

                    // Three measured runs
                    for (int run = 0; run < 3; run++) {

                        auto start = std::chrono::high_resolution_clock::now();

                        result = apply_filter(source, f, impl);

                        auto end = std::chrono::high_resolution_clock::now();

                        total_time +=
                            std::chrono::duration<double, std::milli>(
                                end - start
                            ).count();
                    }

                    double average_time = total_time / 3.0;

                    benchmark_times_ms[f][impl] = average_time;

                    if (impl == 0) {
                        serial_baseline_ms[f] = average_time;
                    }
                }

                // OpenMP and Tiled + OpenMP
                else {

                    // Test thread counts 1 through 8
                    for (int t = 1; t <= 8; t++) {

                        #ifdef _OPENMP
                        omp_set_num_threads(t);
                        #endif

                        Image result;

                        // Warm-up run
                        result = apply_filter(source, f, impl);

                        double total_time = 0.0;

                        // Three measured runs
                        for (int run = 0; run < 3; run++) {

                            auto start = std::chrono::high_resolution_clock::now();

                            result = apply_filter(source, f, impl);

                            auto end = std::chrono::high_resolution_clock::now();

                            total_time +=
                                std::chrono::duration<double, std::milli>(
                                    end - start
                                ).count();
                        }

                        double average_time = total_time / 3.0;

                        thread_benchmark_times_ms[f][impl][t - 1] =
                            average_time;

                        // Use the 4-thread result for the comparison graphs
                        if (t == 4) {
                            benchmark_times_ms[f][impl] = average_time;
                        }
                    }
                }
            }
        }

        // Benchmark different image resolutions
resolution_measurement_count = 0;

double scale_factors[] = {0.25, 0.50, 0.75, 1.00};

#ifdef _OPENMP
omp_set_num_threads(selected_threads);
#endif

for (double scale_factor : scale_factors) {

    int test_width = std::max(
        1,
        static_cast<int>(source.width * scale_factor)
    );

    int test_height = std::max(
        1,
        static_cast<int>(source.height * scale_factor)
    );

    Image test_image = resize_image_nearest(
        source,
        test_width,
        test_height
    );

    // Warm-up run
    Image result = apply_filter(
        test_image,
        selected_filter,
        selected_implementation
    );

    double total_time = 0.0;

    // Three measured runs
    for (int run = 0; run < 3; run++) {

        auto start = std::chrono::high_resolution_clock::now();

        result = apply_filter(
            test_image,
            selected_filter,
            selected_implementation
        );

        auto end = std::chrono::high_resolution_clock::now();

        total_time +=
            std::chrono::duration<double, std::milli>(
                end - start
            ).count();
    }

    resolution_widths[resolution_measurement_count] = test_width;

    resolution_times_ms[resolution_measurement_count] =
        total_time / 3.0;

    resolution_measurement_count++;
}



        // Restore the user's selected thread count
        #ifdef _OPENMP
        omp_set_num_threads(selected_threads);
        #endif

        std::cout << "All benchmarks completed successfully!\n";

    } else {

        std::cout << "Please load an image before running benchmarks.\n";
    }
}

               

               ImGui::Spacing();











                ImGui::Text("Select Filter:");

                ImGui::Combo(
                     "##PerformanceFilter",
                     &selected_filter,
                     filters,
                     IM_ARRAYSIZE(filters)
                );

                ImGui::Separator();











                // First row of graphs
                if (ImGui::BeginTable("GraphRow1", 2)) {

                    ImGui::TableNextColumn();





                    
double implementation_positions[] = {0, 1, 2, 3};

const char* implementation_labels[] = {
    "Serial",
    "Cache-Friendly",
    "OpenMP",
    "Tiled + OpenMP"
};

if (ImPlot::BeginPlot(
    "Execution Time Comparison",
    ImVec2(-1, 280)
)) {
    ImPlot::SetupAxes(
        "Implementation",
        "Time (ms)"
    );

    double max_execution_time = 0.0;

for (int i = 0; i < 4; i++) {
    if (benchmark_times_ms[selected_filter][i] > max_execution_time) {
        max_execution_time = benchmark_times_ms[selected_filter][i];
    }
}

ImPlot::SetupAxisLimits(
    ImAxis_Y1,
    0.0,
    max_execution_time * 1.25,
    ImGuiCond_Always
);



    ImPlot::SetupAxisTicks(
        ImAxis_X1,
        implementation_positions,
        4,
        implementation_labels
    );

    ImPlot::PlotBars(
        "Execution Time",
        implementation_positions,
        benchmark_times_ms[selected_filter],
        4,
        0.6
    );

for (int i = 0; i < 4; i++) {

    double measured_time =
        benchmark_times_ms[selected_filter][i];

    if (measured_time > 0.0) {

        char label[32];

        snprintf(
            label,
            sizeof(label),
            "%.3f ms",
            measured_time
        );

        ImPlot::PlotText(
            label,
            implementation_positions[i],
            measured_time
        );
    }
}



    ImPlot::EndPlot();
}


                    ImGui::TableNextColumn();

                    
double speedup_implementation_positions[] = {0, 1, 2, 3};

const char* speedup_implementation_labels[] = {
    "Serial",
    "Cache-Friendly",
    "OpenMP",
    "Tiled + OpenMP"
};

double implementation_speedups[4] = {};

double serial_time = benchmark_times_ms[selected_filter][0];

if (serial_time > 0.0) {
    for (int i = 0; i < 4; i++) {
        if (benchmark_times_ms[selected_filter][i] > 0.0) {
            implementation_speedups[i] =
                serial_time / benchmark_times_ms[selected_filter][i];
        }
    }
}

if (ImPlot::BeginPlot(
    "Speedup by Implementation",
    ImVec2(-1, 280)
)) {
    ImPlot::SetupAxes(
        "Implementation",
        "Speedup (x)"
    );

    ImPlot::SetupAxisTicks(
        ImAxis_X1,
        speedup_implementation_positions,
        4,
        speedup_implementation_labels
    );

    double max_speedup = 0.0;

for (int i = 0; i < 4; i++) {
    if (implementation_speedups[i] > max_speedup) {
        max_speedup = implementation_speedups[i];
    }
}

ImPlot::SetupAxisLimits(
    ImAxis_Y1,
    0.0,
    max_speedup * 1.25,
    ImGuiCond_Always
);



    ImPlot::PlotBars(
        "Speedup",
        speedup_implementation_positions,
        implementation_speedups,
        4,
        0.6
    );


for (int i = 0; i < 4; i++) {

    if (implementation_speedups[i] > 0.0) {

        char label[32];

        snprintf(
            label,
            sizeof(label),
            "%.2fx",
            implementation_speedups[i]
        );

        ImPlot::PlotText(
            label,
            speedup_implementation_positions[i],
            implementation_speedups[i]
        );
    }
}


    ImPlot::EndPlot();
}


                    ImGui::EndTable();
                }

                // Second row of graphs
                if (ImGui::BeginTable("GraphRow2", 2)) {

                    ImGui::TableNextColumn();

                    if (ImPlot::BeginPlot(
                        "Image Resolution vs Execution Time",
                        ImVec2(-1, 280)
                    )) {
                        ImPlot::SetupAxes(
                            "Image Width (pixels)",
                            "Time (ms)"
                        );

                        if (resolution_measurement_count > 0) {
    double min_width = resolution_widths[0];
    double max_width = resolution_widths[0];
    double max_time = 0.0;

    for (int i = 0; i < resolution_measurement_count; i++) {
        if (resolution_widths[i] < min_width)
            min_width = resolution_widths[i];

        if (resolution_widths[i] > max_width)
            max_width = resolution_widths[i];

        if (resolution_times_ms[i] > max_time)
            max_time = resolution_times_ms[i];
    }

    ImPlot::SetupAxisLimits(
        ImAxis_X1,
        min_width * 0.85,
        max_width * 1.15,
        ImGuiCond_Always
    );

    ImPlot::SetupAxisLimits(
        ImAxis_Y1,
        0.0,
        max_time * 1.25,
        ImGuiCond_Always
    );
}

                        ImPlot::PlotLine(
                            "Execution Time",
                            resolution_widths,
                            resolution_times_ms,
                            resolution_measurement_count
                        );


for (int i = 0; i < resolution_measurement_count; i++) {
    char label[32];
    snprintf(label, sizeof(label), "%.3f ms", resolution_times_ms[i]);

    ImVec2 offset(0, 0);

    if (i == 0) {
        offset = ImVec2(35, 0);
    }
    else if (i == resolution_measurement_count - 1) {
        offset = ImVec2(-35, 0);
    }

    ImPlot::PlotText(
        label,
        resolution_widths[i],
        resolution_times_ms[i],
        offset
    );
}

                        ImPlot::EndPlot();
                    }

                    ImGui::TableNextColumn();

                   
double efficiency_thread_counts[] = {
    1, 2, 3, 4, 5, 6, 7, 8
};

double openmp_efficiency[8] = {};
double tiled_efficiency[8] = {};

double serial_time_for_efficiency =
    serial_baseline_ms[selected_filter];

if (serial_time_for_efficiency > 0.0) {

    for (int i = 0; i < 8; i++) {

        double openmp_time =
            thread_benchmark_times_ms[selected_filter][2][i];

        double tiled_time =
            thread_benchmark_times_ms[selected_filter][3][i];

        if (openmp_time > 0.0) {
            openmp_efficiency[i] =
                (serial_time_for_efficiency / openmp_time)
                / (i + 1) * 100.0;
        }

        if (tiled_time > 0.0) {
            tiled_efficiency[i] =
                (serial_time_for_efficiency / tiled_time)
                / (i + 1) * 100.0;
        }
    }
}

if (ImPlot::BeginPlot(
    "Parallel Efficiency",
    ImVec2(-1, 280)
)) {
    ImPlot::SetupAxes(
        "Threads",
        "Efficiency (%)"
    );

    double max_efficiency = 0.0;

for (int i = 0; i < 8; i++) {
    if (openmp_efficiency[i] > max_efficiency)
        max_efficiency = openmp_efficiency[i];

    if (tiled_efficiency[i] > max_efficiency)
        max_efficiency = tiled_efficiency[i];
}

ImPlot::SetupAxisLimits(
    ImAxis_X1,
    0.5,
    8.5,
    ImGuiCond_Always
);

ImPlot::SetupAxisLimits(
    ImAxis_Y1,
    0.0,
    max_efficiency * 1.25,
    ImGuiCond_Always
);




    ImPlot::PlotLine(
        "OpenMP",
        efficiency_thread_counts,
        openmp_efficiency,
        8
    );

    ImPlot::PlotLine(
        "Tiled + OpenMP",
        efficiency_thread_counts,
        tiled_efficiency,
        8
    );
    for (int i = 0; i < 8; i++) {
    if (openmp_efficiency[i] > 0.0) {
        char label[32];
        snprintf(label, sizeof(label), "%.1f%%", openmp_efficiency[i]);
        ImPlot::PlotText(
            label,
            efficiency_thread_counts[i],
            openmp_efficiency[i] + 2.0
        );
    }

    if (tiled_efficiency[i] > 0.0) {
        char label[32];
        snprintf(label, sizeof(label), "%.1f%%", tiled_efficiency[i]);
        ImPlot::PlotText(
            label,
            efficiency_thread_counts[i],
            tiled_efficiency[i] + 2.0
        );
    }
}


    ImPlot::EndPlot();
}


                    ImGui::EndTable();
                }

                ImGui::EndTabItem();
            }

            // SETTINGS TAB
            if (ImGui::BeginTabItem("Settings")) {

                ImGui::Spacing();
                ImGui::Text("APPLICATION SETTINGS");
                ImGui::Separator();

                ImGui::Text("Default Thread Count");

                ImGui::SliderInt(
                    "Threads",
                    &selected_threads,
                    1,
                    8
                );

                ImGui::Text("Available Filters");
                ImGui::BulletText("Grayscale");
                ImGui::BulletText("Gaussian Blur");
                ImGui::BulletText("Sobel Edge Detection");

                ImGui::Spacing();
                ImGui::Text("Available Implementations");
                ImGui::BulletText("Serial");
                ImGui::BulletText("Cache-Friendly");
                ImGui::BulletText("OpenMP");
                ImGui::BulletText("Tiled + OpenMP");

                ImGui::EndTabItem();
            }

            ImGui::EndTabBar();
        }

        ImGui::End();

        ImGui::Render();

        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);

        glViewport(0, 0, display_w, display_h);
        glClearColor(0.08f, 0.09f, 0.12f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();

    ImPlot::DestroyContext();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
