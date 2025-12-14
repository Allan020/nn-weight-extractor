/**
 * INT16 quantization implementation
 */

#include "int16_quantizer.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <climits>
#include <limits>

namespace darknet {

Int16Quantizer::Int16Quantizer(bool verbose) : verbose_(verbose) {
    init_q_ranges();
}

Int16Quantizer::~Int16Quantizer() {
}

void Int16Quantizer::init_q_ranges() {
    // Initialize Q-format ranges for Q0 to Q15
    // For Q-k: range = [-32768 * 2^(-k), 32767 * 2^(-k)]
    const int16_t ap16_min = INT16_MIN;  // -32768
    const int16_t ap16_max = INT16_MAX;  // 32767
    
    for (int q = 0; q <= MAX_Q; q++) {
        q_ranges_[q][0] = static_cast<float>(ap16_min) * std::pow(2.0, -q);  // min
        q_ranges_[q][1] = static_cast<float>(ap16_max) * std::pow(2.0, -q);  // max
    }
}

std::pair<float, float> Int16Quantizer::get_q_range(int q) const {
    if (q < 0 || q > MAX_Q) {
        return std::make_pair(0.0f, 0.0f);
    }
    return std::make_pair(q_ranges_[q][0], q_ranges_[q][1]);
}

int Int16Quantizer::find_max_q(float min, float max) const {
    // Find the highest Q value (0-15) where min and max fit in the range
    // Test from Q15 (highest precision) down to Q0 (lowest precision)
    for (int q = MAX_Q; q >= 0; q--) {
        float range_min = q_ranges_[q][0];
        float range_max = q_ranges_[q][1];
        
        if (min >= range_min && max <= range_max) {
            return q;
        }
    }
    
    // If no Q fits, values exceed Q0 range
    if (verbose_) {
        std::cerr << "Warning: Values exceed Q0 range (min=" << min 
                  << ", max=" << max << "), using Q0" << std::endl;
    }
    return 0;  // Use Q0 as fallback
}

QuantizedData Int16Quantizer::quantize_layer(
    const std::vector<float>& weights,
    const std::vector<float>& biases
) {
    QuantizedData result;
    
    if (weights.empty() && biases.empty()) {
        return result;
    }
    
    // Find min/max across both weights and biases
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();
    
    // Check weights
    for (float w : weights) {
        if (w < min_val) min_val = w;
        if (w > max_val) max_val = w;
    }
    
    // Check biases
    for (float b : biases) {
        if (b < min_val) min_val = b;
        if (b > max_val) max_val = b;
    }
    
    // Find the maximum Q value that can represent all values
    int max_q = find_max_q(min_val, max_val);
    result.q_value = max_q;
    
    if (verbose_) {
        std::cout << "  Quantization: min=" << min_val << ", max=" << max_val 
                  << ", Q=" << max_q << std::endl;
    }
    
    // Quantize weights: int16_value = (int16_t)(float_value * 2^Q)
    double scale = std::pow(2.0, max_q);
    result.weights.resize(weights.size());
    
    double sum_error = 0.0;
    double max_error = 0.0;
    double min_error = std::numeric_limits<double>::max();
    
    for (size_t i = 0; i < weights.size(); i++) {
        float float_val = weights[i];
        int16_t int16_val = static_cast<int16_t>(float_val * scale);
        
        // Clamp to int16_t range
        if (int16_val > INT16_MAX) int16_val = INT16_MAX;
        if (int16_val < INT16_MIN) int16_val = INT16_MIN;
        
        // Calculate quantization error
        float dequantized = static_cast<float>(int16_val) * std::pow(2.0, -max_q);
        double error = std::abs(dequantized - float_val);
        sum_error += error * error;
        if (error > max_error) max_error = error;
        if (error < min_error) min_error = error;
        
        result.weights[i] = int16_val;
    }
    
    // Quantize biases
    result.biases.resize(biases.size());
    for (size_t i = 0; i < biases.size(); i++) {
        float float_val = biases[i];
        int16_t int16_val = static_cast<int16_t>(float_val * scale);
        
        // Clamp to int16_t range
        if (int16_val > INT16_MAX) int16_val = INT16_MAX;
        if (int16_val < INT16_MIN) int16_val = INT16_MIN;
        
        // Calculate quantization error
        float dequantized = static_cast<float>(int16_val) * std::pow(2.0, -max_q);
        double error = std::abs(dequantized - float_val);
        sum_error += error * error;
        if (error > max_error) max_error = error;
        if (error < min_error) min_error = error;
        
        result.biases[i] = int16_val;
    }
    
    if (verbose_) {
        double rms_error = std::sqrt(sum_error / (weights.size() + biases.size()));
        std::cout << "  Quantization error: RMS=" << rms_error 
                  << ", min=" << min_error << ", max=" << max_error << std::endl;
    }
    
    return result;
}

} // namespace darknet

