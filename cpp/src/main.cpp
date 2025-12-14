/**
 * Darknet weights to binary format extractor
 * 
 * Extracts weights and biases from Darknet .weights files and converts them
 * to binary format with batch normalization folding for FPGA acceleration.
 */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstring>
#include <iomanip>
#include <sstream>

#include "cfg_parser.h"
#include "weights_reader.h"
#include "batch_norm_folder.h"

using namespace darknet;

// Command-line arguments structure
struct Arguments {
    std::string cfg_path;
    std::string weights_path;
    std::string output_weights = "weights.bin";
    std::string output_bias = "bias.bin";
    bool verbose = false;
    bool help = false;
};

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n";
    std::cout << "Extract weights and biases from Darknet format to binary files\n\n";
    std::cout << "Required arguments:\n";
    std::cout << "  --cfg PATH              Path to Darknet .cfg file\n";
    std::cout << "  --weights PATH          Path to Darknet .weights file\n\n";
    std::cout << "Optional arguments:\n";
    std::cout << "  --output-weights PATH   Output weights file (default: weights.bin)\n";
    std::cout << "  --output-bias PATH      Output bias file (default: bias.bin)\n";
    std::cout << "  --verbose, -v           Enable verbose output\n";
    std::cout << "  --help, -h              Show this help message\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << program_name << " --cfg yolov2.cfg --weights yolov2.weights\n";
    std::cout << "  " << program_name << " --cfg model.cfg --weights model.weights --verbose\n";
    std::cout << "  " << program_name << " --cfg model.cfg --weights model.weights \\\n";
    std::cout << "      --output-weights out_w.bin --output-bias out_b.bin\n";
}

Arguments parse_arguments(int argc, char** argv) {
    Arguments args;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            args.help = true;
        } else if (arg == "--verbose" || arg == "-v") {
            args.verbose = true;
        } else if (arg == "--cfg" && i + 1 < argc) {
            args.cfg_path = argv[++i];
        } else if (arg == "--weights" && i + 1 < argc) {
            args.weights_path = argv[++i];
        } else if (arg == "--output-weights" && i + 1 < argc) {
            args.output_weights = argv[++i];
        } else if (arg == "--output-bias" && i + 1 < argc) {
            args.output_bias = argv[++i];
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
        }
    }
    
    return args;
}

bool validate_arguments(const Arguments& args) {
    if (args.help) {
        return false;
    }
    
    if (args.cfg_path.empty()) {
        std::cerr << "Error: --cfg argument is required\n";
        return false;
    }
    
    if (args.weights_path.empty()) {
        std::cerr << "Error: --weights argument is required\n";
        return false;
    }
    
    return true;
}

std::string format_bytes(size_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB"};
    double value = static_cast<double>(bytes);
    int unit_index = 0;
    
    while (value >= 1024.0 && unit_index < 3) {
        value /= 1024.0;
        unit_index++;
    }
    
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(value >= 10.0 ? 1 : 2)
        << value << " " << units[unit_index];
    return oss.str();
}

int main(int argc, char** argv) {
    // Parse command-line arguments
    Arguments args = parse_arguments(argc, argv);
    
    if (args.help) {
        print_usage(argv[0]);
        return 0;
    }
    
    if (!validate_arguments(args)) {
        print_usage(argv[0]);
        return 1;
    }
    
    std::cout << "Darknet Weights Extractor" << std::endl;
    std::cout << "=========================" << std::endl;
    std::cout << std::endl;
    
    // Parse configuration file
    std::cout << "Parsing configuration file: " << args.cfg_path << std::endl;
    CfgParser cfg_parser(args.verbose);
    
    if (!cfg_parser.parse(args.cfg_path)) {
        std::cerr << "Error: Failed to parse config file" << std::endl;
        return 1;
    }
    
    auto conv_layers = cfg_parser.get_conv_layers();
    std::cout << "Found " << conv_layers.size() << " convolutional layers" << std::endl;
    
    if (args.verbose) {
        cfg_parser.print_summary();
    }
    
    // Open weights file
    std::cout << "\nReading weights file: " << args.weights_path << std::endl;
    WeightsReader weights_reader(args.verbose);
    
    if (!weights_reader.open(args.weights_path)) {
        std::cerr << "Error: Failed to open weights file" << std::endl;
        return 1;
    }
    
    // Read header
    weights_reader.read_header();
    
    // Open output files
    std::ofstream weights_out(args.output_weights, std::ios::binary);
    std::ofstream bias_out(args.output_bias, std::ios::binary);
    
    if (!weights_out.is_open()) {
        std::cerr << "Error: Cannot open output weights file: " << args.output_weights << std::endl;
        return 1;
    }
    
    if (!bias_out.is_open()) {
        std::cerr << "Error: Cannot open output bias file: " << args.output_bias << std::endl;
        return 1;
    }
    
    std::cout << "\nExtracting and processing layers..." << std::endl;
    
    // Create batch norm folder
    BatchNormFolder bn_folder;
    
    // Process each convolutional layer
    size_t total_weights = 0;
    size_t total_biases = 0;
    
    for (size_t i = 0; i < conv_layers.size(); i++) {
        const LayerConfig& layer = conv_layers[i];
        
        if (args.verbose) {
            std::cout << "\nLayer " << (i+1) << "/" << conv_layers.size() 
                      << ": " << layer.type << std::endl;
            std::cout << "  Filters: " << layer.filters << std::endl;
            std::cout << "  Size: " << layer.size << "x" << layer.size << std::endl;
            std::cout << "  Stride: " << layer.stride << std::endl;
            std::cout << "  Batch normalize: " << (layer.batch_normalize ? "yes" : "no") << std::endl;
        }
        
        int n = layer.filters;
        int c = layer.channels;
        int num_weights = (c / layer.groups) * n * layer.size * layer.size;
        
        // Read biases
        auto biases = weights_reader.read_biases(n);
        if (biases.size() != (size_t)n) {
            std::cerr << "Error: Failed to read biases for layer " << i << std::endl;
            return 1;
        }
        
        // Read batch normalization parameters if present
        std::vector<float> scales, means, variances;
        if (layer.batch_normalize) {
            scales = weights_reader.read_scales(n);
            means = weights_reader.read_mean(n);
            variances = weights_reader.read_variance(n);
            
            if (scales.size() != (size_t)n || means.size() != (size_t)n || 
                variances.size() != (size_t)n) {
                std::cerr << "Error: Failed to read BN parameters for layer " << i << std::endl;
                return 1;
            }
        }
        
        // Read weights
        auto weights = weights_reader.read_weights(num_weights);
        if (weights.size() != (size_t)num_weights) {
            std::cerr << "Error: Failed to read weights for layer " << i << std::endl;
            return 1;
        }
        
        // Fold batch normalization
        FoldedWeights folded = bn_folder.fold(
            weights, biases, scales, means, variances,
            n, num_weights / n, layer.batch_normalize
        );
        
        // Write to output files
        weights_out.write(reinterpret_cast<const char*>(folded.weights.data()), 
                         folded.weights.size() * sizeof(float));
        bias_out.write(reinterpret_cast<const char*>(folded.biases.data()),
                      folded.biases.size() * sizeof(float));
        
        total_weights += folded.weights.size();
        total_biases += folded.biases.size();

        if (args.verbose) {
            // Detailed per-layer summary
            size_t weights_bytes = folded.weights.size() * sizeof(float);
            size_t bias_bytes = folded.biases.size() * sizeof(float);
            int in_channels = layer.channels;
            int out_channels = layer.filters;
            int kernel = layer.size;
            int groups = layer.groups;
            int in_per_group = (groups > 0) ? (in_channels / groups) : in_channels;

            std::cout << "Layer " << (i+1) << "/" << conv_layers.size()
                      << " [conv]"
                      << " out_ch=" << out_channels
                      << " in_ch=" << in_channels
                      << " groups=" << groups
                      << " kernel=" << kernel << "x" << kernel
                      << " (per-group in=" << in_per_group << ")"
                      << " weights=" << folded.weights.size() << " (" << format_bytes(weights_bytes) << ")"
                      << " bias=" << folded.biases.size() << " (" << format_bytes(bias_bytes) << ")"
                      << (layer.batch_normalize ? " BN-folded" : "")
                      << std::endl;
        } else {
            // Compact progress for non-verbose runs
            std::cout << "\rProcessed layer " << (i+1) << "/" << conv_layers.size() << std::flush;
        }
    }

    if (!args.verbose) {
        std::cout << std::endl;
    }
    
    // Close files
    weights_reader.close();
    weights_out.close();
    bias_out.close();
    
    // Print summary
    std::cout << "\n=========================" << std::endl;
    std::cout << "Extraction completed successfully!" << std::endl;
    std::cout << "=========================" << std::endl;
    std::cout << "\nStatistics:" << std::endl;
    std::cout << "  Layers processed: " << conv_layers.size() << std::endl;
    std::cout << "  Total weights: " << total_weights << std::endl;
    std::cout << "  Total biases: " << total_biases << std::endl;
    std::cout << "  Total parameters: " << (total_weights + total_biases) << std::endl;
    std::cout << "\nOutput files:" << std::endl;
    std::cout << "  Weights: " << args.output_weights << std::endl;
    std::cout << "  Biases:  " << args.output_bias << std::endl;
    
    return 0;
}
