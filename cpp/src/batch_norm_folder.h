/**
 * Batch normalization folding into convolutional weights and biases
 */

#ifndef BATCH_NORM_FOLDER_H
#define BATCH_NORM_FOLDER_H

#include <vector>

namespace darknet {

/**
 * Folded weights and biases structure
 */
struct FoldedWeights {
    std::vector<float> weights;
    std::vector<float> biases;
    
    FoldedWeights() {}
    FoldedWeights(const std::vector<float>& w, const std::vector<float>& b)
        : weights(w), biases(b) {}
};

/**
 * Batch normalization folder
 * 
 * Folds batch normalization parameters into convolutional weights and biases
 * for efficient inference.
 * 
 * Formula:
 * - alpha = scale / sqrt(variance + epsilon)
 * - new_bias = bias - mean * alpha
 * - new_weight = weight * alpha (per output channel)
 */
class BatchNormFolder {
public:
    BatchNormFolder(float epsilon = 0.000001f);
    ~BatchNormFolder();
    
    /**
     * Fold batch normalization into convolutional layer
     * 
     * @param weights Convolutional weights [filters, channels, height, width]
     * @param biases Convolutional biases [filters]
     * @param scales Batch norm scales (gamma) [filters]
     * @param means Batch norm means [filters]
     * @param variances Batch norm variances [filters]
     * @param filters Number of output filters
     * @param has_bn Whether batch normalization is present
     * @return Folded weights and biases
     */
    FoldedWeights fold_conv_layer(
        const std::vector<float>& weights,
        const std::vector<float>& biases,
        const std::vector<float>& scales,
        const std::vector<float>& means,
        const std::vector<float>& variances,
        int filters,
        bool has_bn
    );
    
    /**
     * Fold batch normalization (alternative interface)
     * 
     * @param weights Convolutional weights
     * @param biases Convolutional biases
     * @param scales Batch norm scales
     * @param means Batch norm means
     * @param variances Batch norm variances
     * @param filters Number of output filters
     * @param weights_per_filter Number of weights per filter
     * @param has_bn Whether batch normalization is present
     * @return Folded weights and biases
     */
    FoldedWeights fold(
        const std::vector<float>& weights,
        const std::vector<float>& biases,
        const std::vector<float>& scales,
        const std::vector<float>& means,
        const std::vector<float>& variances,
        int filters,
        int weights_per_filter,
        bool has_bn
    );
    
    /**
     * Set epsilon value for numerical stability
     */
    void set_epsilon(float epsilon) { epsilon_ = epsilon; }
    
    /**
     * Get epsilon value
     */
    float get_epsilon() const { return epsilon_; }

private:
    float epsilon_;  // Small constant for numerical stability
};

} // namespace darknet

#endif // BATCH_NORM_FOLDER_H

