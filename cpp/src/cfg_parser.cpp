/**
 * Darknet configuration file parser implementation
 */

#include "cfg_parser.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cctype>

namespace darknet {

CfgParser::CfgParser(bool verbose) : verbose_(verbose) {
}

CfgParser::~CfgParser() {
}

bool CfgParser::parse(const std::string& cfg_path) {
    std::ifstream file(cfg_path);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open config file: " << cfg_path << std::endl;
        return false;
    }
    
    layers_.clear();
    
    std::string line;
    std::string current_section;
    std::map<std::string, std::string> current_options;
    int layer_index = 0;
    
    while (std::getline(file, line)) {
        // Remove comments
        size_t comment_pos = line.find('#');
        if (comment_pos != std::string::npos) {
            line = line.substr(0, comment_pos);
        }
        
        // Trim whitespace
        line = trim(line);
        
        // Skip empty lines
        if (line.empty()) {
            continue;
        }
        
        // Check for section header [section_name]
        if (line[0] == '[' && line[line.length()-1] == ']') {
            // Process previous section
            if (!current_section.empty()) {
                if (current_section == "net" || current_section == "network") {
                    parse_net_section(current_options);
                } else if (current_section == "convolutional") {
                    layers_.push_back(parse_convolutional(current_options, layer_index++));
                } else if (current_section == "maxpool") {
                    layers_.push_back(parse_maxpool(current_options, layer_index++));
                } else if (current_section == "route") {
                    layers_.push_back(parse_route(current_options, layer_index++));
                } else if (current_section == "shortcut") {
                    layers_.push_back(parse_shortcut(current_options, layer_index++));
                } else if (current_section == "upsample") {
                    layers_.push_back(parse_upsample(current_options, layer_index++));
                } else if (current_section == "yolo" || current_section == "region") {
                    layers_.push_back(parse_yolo(current_options, layer_index++));
                }
            }
            
            // Start new section
            current_section = line.substr(1, line.length()-2);
            current_options.clear();
        } else {
            // Parse key=value option
            size_t eq_pos = line.find('=');
            if (eq_pos != std::string::npos) {
                std::string key = trim(line.substr(0, eq_pos));
                std::string value = trim(line.substr(eq_pos + 1));
                current_options[key] = value;
            }
        }
    }
    
    // Process last section
    if (!current_section.empty()) {
        if (current_section == "net" || current_section == "network") {
            parse_net_section(current_options);
        } else if (current_section == "convolutional") {
            layers_.push_back(parse_convolutional(current_options, layer_index++));
        } else if (current_section == "maxpool") {
            layers_.push_back(parse_maxpool(current_options, layer_index++));
        } else if (current_section == "route") {
            layers_.push_back(parse_route(current_options, layer_index++));
        } else if (current_section == "shortcut") {
            layers_.push_back(parse_shortcut(current_options, layer_index++));
        } else if (current_section == "upsample") {
            layers_.push_back(parse_upsample(current_options, layer_index++));
        } else if (current_section == "yolo" || current_section == "region") {
            layers_.push_back(parse_yolo(current_options, layer_index++));
        }
    }
    
    file.close();
    
    // Infer input/output channel counts for each layer so weight sizes match the model graph
    compute_layer_channels();
    
    if (verbose_) {
        std::cout << "Parsed " << layers_.size() << " layers from config file" << std::endl;
    }
    
    return true;
}

void CfgParser::parse_net_section(const std::map<std::string, std::string>& options) {
    net_config_.width = get_int_option(options, "width", 416);
    net_config_.height = get_int_option(options, "height", 416);
    net_config_.channels = get_int_option(options, "channels", 3);
    net_config_.batch = get_int_option(options, "batch", 1);
    net_config_.subdivisions = get_int_option(options, "subdivisions", 1);
    net_config_.momentum = get_float_option(options, "momentum", 0.9);
    net_config_.decay = get_float_option(options, "decay", 0.0005);
    
    if (verbose_) {
        std::cout << "Network config: " << net_config_.width << "x" << net_config_.height 
                  << "x" << net_config_.channels << std::endl;
    }
}

LayerConfig CfgParser::parse_convolutional(const std::map<std::string, std::string>& options, int index) {
    LayerConfig layer;
    layer.type = "convolutional";
    layer.index = index;
    layer.filters = get_int_option(options, "filters", 1);
    layer.size = get_int_option(options, "size", 1);
    layer.stride = get_int_option(options, "stride", 1);
    layer.pad = get_int_option(options, "pad", 0);
    layer.groups = get_int_option(options, "groups", 1);
    layer.batch_normalize = has_option(options, "batch_normalize");
    layer.activation = get_string_option(options, "activation", "linear");
    
    // Calculate channels from previous layer (will be updated during processing)
    layer.channels = net_config_.channels;
    
    return layer;
}

LayerConfig CfgParser::parse_maxpool(const std::map<std::string, std::string>& options, int index) {
    LayerConfig layer;
    layer.type = "maxpool";
    layer.index = index;
    layer.pool_size = get_int_option(options, "size", 2);
    layer.pool_stride = get_int_option(options, "stride", 2);
    
    return layer;
}

LayerConfig CfgParser::parse_route(const std::map<std::string, std::string>& options, int index) {
    LayerConfig layer;
    layer.type = "route";
    layer.index = index;
    
    std::string layers_str = get_string_option(options, "layers", "");
    if (!layers_str.empty()) {
        layer.layers = parse_int_list(layers_str);
    }
    
    return layer;
}

LayerConfig CfgParser::parse_shortcut(const std::map<std::string, std::string>& options, int index) {
    LayerConfig layer;
    layer.type = "shortcut";
    layer.index = index;
    
    int from = get_int_option(options, "from", -3);
    layer.layers.push_back(from);
    layer.activation = get_string_option(options, "activation", "linear");
    
    return layer;
}

LayerConfig CfgParser::parse_upsample(const std::map<std::string, std::string>& options, int index) {
    LayerConfig layer;
    layer.type = "upsample";
    layer.index = index;
    layer.stride = get_int_option(options, "stride", 2);
    
    return layer;
}

LayerConfig CfgParser::parse_yolo(const std::map<std::string, std::string>& options, int index) {
    (void)options;  // Suppress unused parameter warning
    LayerConfig layer;
    layer.type = "yolo";
    layer.index = index;
    
    return layer;
}

std::vector<LayerConfig> CfgParser::get_conv_layers() const {
    std::vector<LayerConfig> conv_layers;
    for (const auto& layer : layers_) {
        if (layer.is_convolutional()) {
            conv_layers.push_back(layer);
        }
    }
    return conv_layers;
}

const LayerConfig* CfgParser::get_layer(int index) const {
    for (const auto& layer : layers_) {
        if (layer.index == index) {
            return &layer;
        }
    }
    return nullptr;
}

void CfgParser::print_summary() const {
    std::cout << "\nConfiguration Summary:" << std::endl;
    std::cout << "Network: " << net_config_.width << "x" << net_config_.height 
              << "x" << net_config_.channels << std::endl;
    std::cout << "Total layers: " << layers_.size() << std::endl;
    
    int conv_count = 0;
    int maxpool_count = 0;
    int other_count = 0;
    
    for (const auto& layer : layers_) {
        if (layer.is_convolutional()) {
            conv_count++;
        } else if (layer.is_maxpool()) {
            maxpool_count++;
        } else {
            other_count++;
        }
    }
    
    std::cout << "  Convolutional: " << conv_count << std::endl;
    std::cout << "  MaxPool: " << maxpool_count << std::endl;
    std::cout << "  Other: " << other_count << std::endl;
    
    if (verbose_) {
        std::cout << "\nLayer details:" << std::endl;
        for (const auto& layer : layers_) {
            std::cout << "  [" << layer.index << "] " << layer.type;
            if (layer.is_convolutional()) {
                std::cout << " - filters=" << layer.filters 
                         << ", size=" << layer.size 
                         << ", stride=" << layer.stride;
                if (layer.batch_normalize) {
                    std::cout << ", BN";
                }
            }
            std::cout << std::endl;
        }
    }
}

// Helper methods
int CfgParser::get_int_option(const std::map<std::string, std::string>& options,
                             const std::string& key, int default_value) const {
    auto it = options.find(key);
    if (it != options.end()) {
        return std::stoi(it->second);
    }
    return default_value;
}

float CfgParser::get_float_option(const std::map<std::string, std::string>& options,
                                 const std::string& key, float default_value) const {
    auto it = options.find(key);
    if (it != options.end()) {
        return std::stof(it->second);
    }
    return default_value;
}

std::string CfgParser::get_string_option(const std::map<std::string, std::string>& options,
                                        const std::string& key, const std::string& default_value) const {
    auto it = options.find(key);
    if (it != options.end()) {
        return it->second;
    }
    return default_value;
}

bool CfgParser::has_option(const std::map<std::string, std::string>& options,
                          const std::string& key) const {
    return options.find(key) != options.end();
}

std::string CfgParser::trim(const std::string& str) const {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return "";
    }
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, last - first + 1);
}

std::vector<int> CfgParser::parse_int_list(const std::string& str) const {
    std::vector<int> result;
    std::stringstream ss(str);
    std::string item;
    
    while (std::getline(ss, item, ',')) {
        item = trim(item);
        if (!item.empty()) {
            result.push_back(std::stoi(item));
        }
    }
    
    return result;
}

void CfgParser::compute_layer_channels() {
    if (layers_.empty()) {
        return;
    }
    
    std::vector<int> output_channels(layers_.size(), 0);
    int current_channels = net_config_.channels;
    
    for (size_t i = 0; i < layers_.size(); ++i) {
        LayerConfig& layer = layers_[i];
        
        if (layer.is_convolutional()) {
            // Convolutional layers consume the current channel count
            layer.channels = current_channels;
            output_channels[i] = layer.filters;
        } else if (layer.is_route()) {
            // Route concatenates the referenced layers
            int channels_sum = 0;
            for (int layer_ref : layer.layers) {
                int idx = resolve_layer_index(static_cast<int>(i), layer_ref);
                if (idx >= 0 && idx < static_cast<int>(output_channels.size())) {
                    channels_sum += output_channels[idx];
                } else if (verbose_) {
                    std::cerr << "Warning: route layer " << i 
                              << " references invalid layer index " << layer_ref << std::endl;
                }
            }
            output_channels[i] = channels_sum;
        } else if (layer.is_shortcut()) {
            // Shortcut adds the previous output with a referenced layer
            int previous_channels = current_channels;
            int from_idx = layer.layers.empty() ? static_cast<int>(i) - 1
                                               : resolve_layer_index(static_cast<int>(i), layer.layers[0]);
            int from_channels = (from_idx >= 0 && from_idx < static_cast<int>(output_channels.size()))
                ? output_channels[from_idx]
                : previous_channels;
            
            if (previous_channels != from_channels && verbose_) {
                std::cerr << "Warning: shortcut layer " << i 
                          << " channel mismatch (prev=" << previous_channels 
                          << ", from=" << from_channels << ")" << std::endl;
            }
            output_channels[i] = previous_channels;
        } else {
            // Layers without weights keep channel count unchanged
            output_channels[i] = current_channels;
        }
        
        current_channels = output_channels[i];
    }
}

int CfgParser::resolve_layer_index(int current_index, int reference) const {
    // Darknet uses negative values to reference layers relative to the current index
    return (reference >= 0) ? reference : current_index + reference;
}

} // namespace darknet
