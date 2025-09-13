#pragma once

#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <yaml-cpp/yaml.h>

/**
 * @brief Wrapper around ONNX Runtime for Demucs/MDX-Net models.
 *
 * This class mirrors the Python ``separate`` function used throughout the
 * repository.  Audio is separated into stems using an ONNX model chosen by a
 * YAML configuration file.  A progress callback can be supplied to receive
 * updates in the range ``[0, 1]``.
 */
class DemucsMdxRunner {
public:
    using ProgressCallback = std::function<void(float)>;

    /**
     * @param config_path Path to a YAML file describing available models.
     * @param use_gpu     If true, attempt to enable the CUDA execution provider.
     */
    explicit DemucsMdxRunner(const std::string &config_path, bool use_gpu = false);

    /**
     * @brief Separate ``input_path`` into individual stems.
     *
     * @param input_path Path to the input audio file.
     * @param output_dir Directory where stems will be written.
     * @param progress   Optional callback invoked with values in ``[0, 1]``.
     *
     * @return Path to the directory containing the separated stems.
     */
    std::filesystem::path separate(const std::filesystem::path &input_path,
                                   const std::filesystem::path &output_dir,
                                   ProgressCallback progress = nullptr);

private:
    void load_model(const std::string &model_key);
    void run_inference(const std::vector<float> &audio, std::vector<float> &output,
                       ProgressCallback progress);

    Ort::Env env_;
    Ort::SessionOptions session_options_;
    YAML::Node config_;
    std::unique_ptr<Ort::Session> session_;
};

