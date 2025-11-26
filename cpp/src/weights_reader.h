/**
 * Darknet weights file reader
 */

#ifndef WEIGHTS_READER_H
#define WEIGHTS_READER_H

#include <string>
#include <vector>
#include <fstream>
#include <cstdint>

namespace darknet {

/**
 * Weights file header information
 */
struct WeightsHeader {
    int32_t major;
    int32_t minor;
    int32_t revision;
    int64_t seen;
    
    WeightsHeader() : major(0), minor(0), revision(0), seen(0) {}
};

/**
 * Weights file reader
 */
class WeightsReader {
public:
    WeightsReader(bool verbose = false);
    ~WeightsReader();
    
    /**
     * Open weights file for reading
     * @param weights_path Path to .weights file
     * @return true if successful
     */
    bool open(const std::string& weights_path);
    
    /**
     * Close weights file
     */
    void close();
    
    /**
     * Read and parse header
     * @return WeightsHeader structure
     */
    WeightsHeader read_header();
    
    /**
     * Read biases
     * @param n Number of biases to read
     * @return Vector of bias values
     */
    std::vector<float> read_biases(int n);
    
    /**
     * Read batch normalization scales
     * @param n Number of scales to read
     * @return Vector of scale values
     */
    std::vector<float> read_scales(int n);
    
    /**
     * Read batch normalization means
     * @param n Number of means to read
     * @return Vector of mean values
     */
    std::vector<float> read_mean(int n);
    
    /**
     * Read batch normalization variances
     * @param n Number of variances to read
     * @return Vector of variance values
     */
    std::vector<float> read_variance(int n);
    
    /**
     * Read weights
     * @param num Number of weights to read
     * @return Vector of weight values
     */
    std::vector<float> read_weights(int num);
    
    /**
     * Check if file is open
     */
    bool is_open() const { return file_.is_open(); }
    
    /**
     * Get current file position
     */
    size_t tell() const;
    
    /**
     * Get total bytes read
     */
    size_t bytes_read() const { return total_bytes_read_; }
    
private:
    bool verbose_;
    std::ifstream file_;
    size_t total_bytes_read_;
    WeightsHeader header_;
    
    // Helper method to read float array
    std::vector<float> read_floats(int count);
};

} // namespace darknet

#endif // WEIGHTS_READER_H

