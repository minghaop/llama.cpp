//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>

// Windows doesn't get math constants from math headers
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/**
 * @brief CPU Random Number Generator aligned with PyTorch's CPUGeneratorImpl
 *
 * This implementation uses MT19937-64 (Mersenne Twister) with algorithms
 * matching PyTorch's behavior. To achieve exact alignment with PyTorch,
 * the calling code must match PyTorch's exact sequence of RNG calls.
 */
class CPUGenerator {
 public:
  using engine_type = std::mt19937_64;

  CPUGenerator() : CPUGenerator(default_seed()) {}

  /**
   * @brief Construct generator with specific seed (PyTorch-compatible)
   * @param seed The seed value (default PyTorch seed is 5489)
   */
  explicit CPUGenerator(uint64_t seed)
      : engine_(seed), has_next_normal_(false), next_normal_(0.0) {}

  /**
   * @brief Reseed the generator (clears cached normal value)
   * @param seed New seed value
   */
  void reseed(uint64_t seed) {
    engine_.seed(seed);
    has_next_normal_ = false;
    next_normal_     = 0.0;
  }

  /**
   * @brief Generate raw 64-bit unsigned integer
   * @return Random uint64_t value
   */
  uint64_t random_u64() { return engine_(); }

  /**
   * @brief Generate uniform random double in [0, 1)
   * @return Random double in [0, 1)
   *
   * Uses the same method as PyTorch: converts uint64 to double
   * by dividing by 2^64, ensuring exact compatibility.
   */
  double uniform01() {
    // PyTorch-compatible: generate uniform [0,1) from uint64
    // This matches PyTorch's uniform_real_distribution behavior
    constexpr double divisor = static_cast<double>(UINT64_MAX) + 1.0;
    return static_cast<double>(engine_()) / divisor;
  }

  /**
   * @brief Generate uniform random integer in [a, b] (inclusive)
   * @param a Lower bound (inclusive)
   * @param b Upper bound (inclusive)
   * @return Random integer in [a, b]
   *
   * Uses PyTorch's algorithm: generate uniform [0,1) and scale to [a, b]
   * This matches torch::randint behavior exactly.
   */
  template <typename Int>
  Int uniform_int(Int a, Int b) {
    // PyTorch's randint algorithm:
    // 1. Generate uniform double in [0, 1)
    // 2. Scale to [a, b+1) range
    // 3. Floor to get integer in [a, b]
    double u   = uniform01();
    Int range  = b - a + 1;
    Int result = a + static_cast<Int>(u * range);

    // Clamp to ensure we don't exceed b due to floating point precision
    if (result > b) result = b;

    return result;
  }

  /**
   * @brief Generate normal (Gaussian) distributed random number
   * @param mean Mean of the distribution (default 0.0)
   * @param stddev Standard deviation (default 1.0)
   * @return Random value from N(mean, stddev^2)
   *
   * Uses Box-Muller transform with caching (same as PyTorch) to generate
   * pairs of independent normal variates, improving efficiency and ensuring
   * compatibility with PyTorch's normal distribution.
   */
  double normal(double mean = 0.0, double stddev = 1.0) {
    // Use cached value if available (Box-Muller generates pairs)
    if (has_next_normal_) {
      has_next_normal_ = false;
      return mean + stddev * next_normal_;
    }

    // Box-Muller transform (same as PyTorch's implementation)
    // Generate two uniform random numbers
    double u1, u2;
    do {
      u1 = uniform01();
      u2 = uniform01();
    } while (u1 <= std::numeric_limits<double>::epsilon());

    // Apply Box-Muller transform
    const double radius = std::sqrt(-2.0 * std::log(u1));
    const double theta  = 2.0 * M_PI * u2;

    // Generate two independent normal(0,1) variates
    const double z0 = radius * std::cos(theta);
    const double z1 = radius * std::sin(theta);

    // Cache one for next call
    next_normal_     = z1;
    has_next_normal_ = true;

    // Return the other, scaled to desired mean and stddev
    return mean + stddev * z0;
  }

  /**
   * @brief Generate a vector of normal distributed random numbers (tensor-style)
   * @param size Number of values to generate
   * @param mean Mean of the distribution (default 0.0)
   * @param stddev Standard deviation (default 1.0)
   * @return Vector of random values from N(mean, stddev^2)
   *
   * This method matches PyTorch's torch::randn behavior by generating
   * all values in one call, ensuring identical RNG state consumption.
   */
  template <typename T = float>
  std::vector<T> randn(size_t size, double mean = 0.0, double stddev = 1.0) {
    std::vector<T> result(size);
    for (size_t i = 0; i < size; ++i) {
      result[i] = static_cast<T>(normal(mean, stddev));
    }
    return result;
  }

  /**
   * @brief Access underlying MT19937-64 engine
   * @return Reference to the engine for use with std:: distributions
   */
  engine_type& engine() { return engine_; }

  /**
   * @brief Get thread-local singleton instance
   * @param seed Seed value (default 5489 matches PyTorch's default)
   * @return Reference to thread-local CPUGenerator
   */
  static CPUGenerator& thread_local_instance(uint64_t seed = 5489u) {
    thread_local CPUGenerator g(seed);
    return g;
  }

 private:
  /**
   * @brief Generate default seed from random_device
   * @return 64-bit seed value
   */
  static uint64_t default_seed() {
    std::random_device rd;
    // Mix down to 64 bits
    uint64_t s = 0;
    for (int i = 0; i < 4; ++i) {
      s = (s << 16) ^ rd();
    }
    return s;
  }

  engine_type engine_;    ///< MT19937-64 random number engine
  bool has_next_normal_;  ///< Flag for cached normal value
  double next_normal_;    ///< Cached normal value from Box-Muller
};
