#include "DemucsMdxRunner.h"

#include <fstream>
#include <stdexcept>

using namespace std::string_literals;

DemucsMdxRunner::DemucsMdxRunner(const std::string &config_path, bool use_gpu)
    : env_(ORT_LOGGING_LEVEL_WARNING, "demucs_mdx"), session_options_{} {
    if (use_gpu) {
        // Try to enable CUDA provider.  Failure is non-fatal; CPU will be used.
        try {
            OrtSessionOptionsAppendExecutionProvider_CUDA(session_options_, 0);
        } catch (const Ort::Exception &) {
            // Ignore and fall back to CPU
        }
    }
    config_ = YAML::LoadFile(config_path);
}

std::filesystem::path DemucsMdxRunner::separate(const std::filesystem::path &input_path,
                                                const std::filesystem::path &output_dir,
                                                ProgressCallback progress) {
    if (!config_["default_model"]) {
        throw std::runtime_error("No default_model defined in config");
    }
    load_model(config_["default_model"].as<std::string>());

    // TODO: actually load audio file and write separated stems.
    std::vector<float> audio;   // placeholder for input samples
    std::vector<float> results; // placeholder for output
    run_inference(audio, results, progress);

    if (progress)
        progress(1.0f);

    std::filesystem::create_directories(output_dir);
    return output_dir;
}

void DemucsMdxRunner::load_model(const std::string &model_key) {
    auto models = config_["models"];
    if (!models || !models[model_key]) {
        throw std::runtime_error("Model key not found in config: " + model_key);
    }
    auto model_path = models[model_key]["path"].as<std::string>();
    session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);
}

void DemucsMdxRunner::run_inference(const std::vector<float> &audio, std::vector<float> &output,
                                    ProgressCallback progress) {
    if (!session_) {
        throw std::runtime_error("ONNX session not initialised");
    }

    // This is a highly simplified placeholder for actual model inference.  A real
    // implementation would chunk the audio, normalise it, and feed it through the
    // network, invoking the progress callback as chunks are processed.
    if (progress)
        progress(0.5f);

    output = audio; // placeholder copy
}

