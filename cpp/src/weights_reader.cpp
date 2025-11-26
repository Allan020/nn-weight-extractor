/**
 * Darknet weights file reader implementation
 */

#include "weights_reader.h"
#include <iostream>
#include <cstring>

namespace darknet {

WeightsReader::WeightsReader(bool verbose) 
    : verbose_(verbose), total_bytes_read_(0) {
}

WeightsReader::~WeightsReader() {
    close();
}

bool WeightsReader::open(const std::string& weights_path) {
    file_.open(weights_path, std::ios::binary);
    
    if (!file_.is_open()) {
        std::cerr << "Error: Cannot open weights file: " << weights_path << std::endl;
        return false;
    }
    
    total_bytes_read_ = 0;
    
    if (verbose_) {
        std::cout << "Opened weights file: " << weights_path << std::endl;
    }
    
    return true;
}

void WeightsReader::close() {
    if (file_.is_open()) {
        file_.close();
    }
}

WeightsHeader WeightsReader::read_header() {
    if (!file_.is_open()) {
        std::cerr << "Error: File not open" << std::endl;
        return WeightsHeader();
    }
    
    // Read major, minor, revision (3 x int32)
    file_.read(reinterpret_cast<char*>(&header_.major), sizeof(int32_t));
    file_.read(reinterpret_cast<char*>(&header_.minor), sizeof(int32_t));
    file_.read(reinterpret_cast<char*>(&header_.revision), sizeof(int32_t));
    total_bytes_read_ += 3 * sizeof(int32_t);
    
    // Read seen
    // For version >= 0.2, seen is int64, otherwise int32
    if ((header_.major * 10 + header_.minor) >= 2) {
        file_.read(reinterpret_cast<char*>(&header_.seen), sizeof(int64_t));
        total_bytes_read_ += sizeof(int64_t);
    } else {
        int32_t seen32;
        file_.read(reinterpret_cast<char*>(&seen32), sizeof(int32_t));
        header_.seen = seen32;
        total_bytes_read_ += sizeof(int32_t);
    }
    
    if (verbose_) {
        std::cout << "Weights file header:" << std::endl;
        std::cout << "  Version: " << header_.major << "." << header_.minor 
                  << "." << header_.revision << std::endl;
        std::cout << "  Images seen: " << header_.seen << std::endl;
    }
    
    return header_;
}

std::vector<float> WeightsReader::read_biases(int n) {
    return read_floats(n);
}

std::vector<float> WeightsReader::read_scales(int n) {
    return read_floats(n);
}

std::vector<float> WeightsReader::read_mean(int n) {
    return read_floats(n);
}

std::vector<float> WeightsReader::read_variance(int n) {
    return read_floats(n);
}

std::vector<float> WeightsReader::read_weights(int num) {
    return read_floats(num);
}

std::vector<float> WeightsReader::read_floats(int count) {
    if (!file_.is_open()) {
        std::cerr << "Error: File not open" << std::endl;
        return std::vector<float>();
    }
    
    if (count <= 0) {
        return std::vector<float>();
    }
    
    std::vector<float> data(count);
    file_.read(reinterpret_cast<char*>(data.data()), count * sizeof(float));
    
    if (!file_) {
        std::cerr << "Error: Failed to read " << count << " floats from file" << std::endl;
        std::cerr << "  Bytes read: " << file_.gcount() << std::endl;
        std::cerr << "  Expected: " << count * sizeof(float) << std::endl;
        return std::vector<float>();
    }
    
    total_bytes_read_ += count * sizeof(float);
    
    return data;
}

size_t WeightsReader::tell() const {
    if (!file_.is_open()) {
        return 0;
    }
    // Cast away const to call tellg() - this doesn't modify the logical state
    return const_cast<std::ifstream&>(file_).tellg();
}

} // namespace darknet

