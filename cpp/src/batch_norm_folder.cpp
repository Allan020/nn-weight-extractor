/**
 * Batch normalization folding implementation
 */

#include "batch_norm_folder.h"
#include <cmath>
#include <iostream>

namespace darknet {

BatchNormFolder::BatchNormFolder(float epsilon) : epsilon_(epsilon) {
}

BatchNormFolder::~BatchNormFolder() {
}

FoldedWeights BatchNormFolder::fold_conv_layer(
    const std::vector<float>& weights,
    const std::vector<float>& biases,
    const std::vector<float>& scales,
    const std::vector<float>& means,
    const std::vector<float>& variances,
    int filters,
    bool has_bn
) {
    int weights_per_filter = weights.size() / filters;
    return fold(weights, biases, scales, means, variances, 
                filters, weights_per_filter, has_bn);
}

FoldedWeights BatchNormFolder::fold(
    const std::vector<float>& weights,
    const std::vector<float>& biases,
    const std::vector<float>& scales,
    const std::vector<float>& means,
    const std::vector<float>& variances,
    int filters,
    int weights_per_filter,
    bool has_bn
) {
    FoldedWeights result;
    
    // Allocate output buffers
    result.weights.resize(weights.size());
    result.biases.resize(filters);
    
    // Calculate alpha and folded biases
    std::vector<float> alpha(filters);
    
    if (has_bn) {
        // With batch normalization
        // alpha = scale / sqrt(variance + epsilon)
        // folded_bias = bias - mean * alpha
        for (int i = 0; i < filters; i++) {
            float tmp = scales[i] / std::sqrt(variances[i] + epsilon_);
            alpha[i] = tmp;
            result.biases[i] = biases[i] - means[i] * tmp;
        }
    } else {
        // Without batch normalization
        // alpha = 1.0
        // folded_bias = bias
        for (int i = 0; i < filters; i++) {
            alpha[i] = 1.0f;
            result.biases[i] = biases[i];
        }
    }
    
    // Apply alpha to weights
    // Each filter's weights are multiplied by the corresponding alpha value
    int cnt = 0;
    for (int j = 0; j < filters; j++) {
        for (int i = 0; i < weights_per_filter; i++) {
            result.weights[cnt] = weights[cnt] * alpha[j];
            cnt++;
        }
    }
    
    return result;
}

} // namespace darknet

