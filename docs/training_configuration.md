# Training Configuration Comparison

This document compares the experimental configuration used in this reproduction with the original implementation released by Gohr for the paper:

> Gohr, A. (2019). *Improving Attacks on Round-Reduced Speck32/64 Using Deep Learning*. CRYPTO 2019.

## Experimental Configuration

| Parameter | Original Gohr Implementation | This Reproduction | Status |
|-----------|------------------------------|-------------------|--------|
| Cipher | Speck32/64 | Speck32/64 | ✅ Match |
| Number of rounds | 5 | 5 | ✅ Match |
| Network architecture | Residual CNN | Residual CNN | ✅ Match |
| Residual blocks | 10 | 10 | ✅ Match |
| Epochs | 200 | 200 | ✅ Match |
| Optimizer | Adam | Adam | ✅ Match |
| Learning-rate schedule | Cyclic Learning Rate | Cyclic Learning Rate | ✅ Match |
| Loss function | Mean Squared Error (MSE) | Mean Squared Error (MSE) | ✅ Match |
| Batch generation | Online synthetic generation | Online synthetic generation | ✅ Match |
| Training samples per epoch | 10,000,000 | 10,000,000 | ✅ Match |
| Validation samples | 1,000,000 | 1,000,000 | ✅ Match |

## Notes

This reproduction preserves the original architecture and training pipeline. No modifications were made to:

- model architecture
- loss function
- optimizer
- learning-rate schedule
- data generation procedure
- network depth
- training duration

The only changes required were compatibility updates for TensorFlow 2.20 and Keras 3. These changes did not alter the learning algorithm or experimental protocol.