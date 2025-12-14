/**
 * INT16 quantization using Q-format fixed-point representation
 */

#ifndef INT16_QUANTIZER_H
#define INT16_QUANTIZER_H

#include <vector>
#include <cstdint>

namespace darknet {

/**
 * Quantized data structure
 */
struct QuantizedData {
    std::vector<int16_t> weights;
    std::vector<int16_t> biases;
    int q_value;  // Q value (0-15) used for quantization
    
    QuantizedData() : q_value(-1) {}
    QuantizedData(const std::vector<int16_t>& w, const std::vector<int16_t>& b, int q)
        : weights(w), biases(b), q_value(q) {}
};

/**
 * INT16 Quantizer
 * 
 * Converts float32 weights and biases to INT16 format using Q-format
 * fixed-point quantization. Each layer gets its own Q value (0-15) that
 * determines the quantization scale: value_int16 = value_float * 2^Q
 * 
 * Q-format ranges:
 * - Q0: [-32768, 32767]
 * - Q1: [-16384, 16383.5]
 * - Q2: [-8192, 8191.75]
 * - ...
 * - Q15: [-1, 0.99996948]
 */
class Int16Quantizer {
public:
    Int16Quantizer(bool verbose = false);
    ~Int16Quantizer();
    
    /**
     * Quantize a layer's weights and biases
     * 
     * @param weights Layer weights (float32)
     * @param biases Layer biases (float32)
     * @return QuantizedData containing INT16 values and Q value
     */
    QuantizedData quantize_layer(
        const std::vector<float>& weights,
        const std::vector<float>& biases
    );
    
    /**
     * Get Q-format range for a given Q value
     * 
     * @param q Q value (0-15)
     * @return Pair of (min, max) range values
     */
    std::pair<float, float> get_q_range(int q) const;
    
    /**
     * Find the maximum Q value that can represent all values in a range
     * 
     * @param min Minimum value
     * @param max Maximum value
     * @return Q value (0-15), or -1 if values exceed Q0 range
     */
    int find_max_q(float min, float max) const;

private:
    bool verbose_;
    
    // Pre-computed Q-format ranges (Q0-Q15)
    // Each Q value has [min, max] range
    static const int MAX_Q = 15;
    float q_ranges_[MAX_Q + 1][2];  // [Q][0]=min, [Q][1]=max
    
    /**
     * Initialize Q-format ranges
     */
    void init_q_ranges();
};

} // namespace darknet

#endif // INT16_QUANTIZER_H

