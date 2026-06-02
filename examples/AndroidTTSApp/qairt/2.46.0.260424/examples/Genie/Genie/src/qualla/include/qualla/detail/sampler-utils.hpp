//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef QUALLA_DETAIL_SAMPLER_UTILS_HPP
#define QUALLA_DETAIL_SAMPLER_UTILS_HPP

#ifdef _MSC_VER
#pragma warning(disable : 4068)
#endif

#include <deque>
#include <functional>
#include <queue>
#include <random>
#include <span>
#include <string>
#include <utility>

#include "Trace.hpp"
#include "Traceable.hpp"
#include "qualla/detail/preproc.hpp"
#include "qualla/detail/tensor.hpp"
#include "qualla/detail/utils.hpp"

#define TOPP_SAMPLER_INITIAL_PARTITION_POINT 4096

namespace qualla {

// penalty Struct
struct Penalty {
  Penalty(const nlohmann::json& conf) {
    // Parse config
    using qc = qualla::Config;

    m_penaltyLastN   = qc::optional<int32_t>(conf, "penalize-last-n", 0);
    m_penaltyPresent = qc::optional<float>(conf, "presence-penalty", 0.0);
    m_penaltyFreq    = qc::optional<float>(conf, "frequency-penalty", 0.0);
    m_penaltyRepeat  = qc::optional<float>(conf, "repetition-penalty", 0.0);
    m_tokens.clear();
    m_tokenFreqMap.clear();
  }

  Penalty(const Penalty&) = delete;

  Penalty& operator=(const Penalty& other) {
    if (this != &other) {
      m_penaltyLastN   = other.m_penaltyLastN;
      m_penaltyPresent = other.m_penaltyPresent;
      m_penaltyFreq    = other.m_penaltyFreq;
      m_penaltyRepeat  = other.m_penaltyRepeat;
      m_tokenFreqMap   = other.m_tokenFreqMap;
      m_tokens         = other.m_tokens;
    }
    return *this;
  }

  Penalty(Penalty&& other) = delete;

  Penalty& operator=(Penalty&& other) = delete;

  void reset() {
    m_tokens.clear();
    m_tokenFreqMap.clear();
  }

  void updateSampledTokenHistory(int32_t tokenIdx, int32_t streamIdx) {
    if (m_penaltyLastN == 0) return;
    if (m_tokens.size() <= streamIdx) {
      m_tokens.push_back({});
      m_tokenFreqMap.push_back({});
    }
    if (m_penaltyLastN == static_cast<int32_t>(m_tokens[streamIdx].size())) {
      m_tokenFreqMap[streamIdx][m_tokens[streamIdx].front()]--;
      if (m_tokenFreqMap[streamIdx][m_tokens[streamIdx].front()] == 0)
        m_tokenFreqMap[streamIdx].erase(m_tokens[streamIdx].front());
      m_tokens[streamIdx].pop_front();
    }
    m_tokens[streamIdx].push_back(tokenIdx);
    m_tokenFreqMap[streamIdx][tokenIdx]++;
  }

  int32_t m_penaltyLastN;
  std::vector<std::unordered_map<int32_t, int32_t>> m_tokenFreqMap;
  std::vector<std::deque<int32_t>> m_tokens;
  float m_penaltyPresent;
  float m_penaltyFreq;
  float m_penaltyRepeat;
};

typedef std::mt19937 rng_t;

// Various sampling utilities.

static double sampleFromUniform(rng_t& rng) {
  int a         = rng() >> 5;
  int b         = rng() >> 6;
  double sample = (a * 67108864.0 + b) / 9007199254740992.0;
  return sample;
}

static inline double sampleFromGumbel(rng_t& rng) {
  double tiny    = 1.1754943508222875e-38;
  double eps     = 1.1920928955078125e-07;
  double uniform = sampleFromUniform(rng);
  double gumbel  = -std::log(-std::log(tiny + uniform * (1. - eps - tiny)));
  return gumbel;
}

// Returns the index of an element chosen by applying the given probability distribution.
template <typename T>
static int32_t sampleFromProbs(const std::span<T> probs, rng_t& rng) {
  static_assert(std::is_floating_point<T>::value);
  std::discrete_distribution<> dist(probs.begin(), probs.end());
  return dist(rng);
}

// Returns the index of the element chosen by the Gumbel max algorithm
template <typename T>
static int32_t sampleUsingGumbelMax(const std::span<T> log_probs, rng_t& rng) {
  static_assert(std::is_floating_point<T>::value);
  int32_t max_purturbed_logit = std::numeric_limits<int32_t>::min();
  int32_t max_idx             = 0;

  for (size_t i = 0; i < log_probs.size(); i++) {
    float purturbed_logit = log_probs[i] + sampleFromGumbel(rng);
    if (purturbed_logit > max_purturbed_logit) {
      max_purturbed_logit = purturbed_logit;
      max_idx             = static_cast<int32_t>(i);
    }
  }
  return max_idx;
}

// Add gumbel noise to a set of logits
template <typename T>
void addGumbelNoise(std::vector<T>& log_probs, rng_t& rng) {
  static_assert(std::is_floating_point<T>::value);
  for (int32_t i = 0; i < log_probs.size(); i++) {
    log_probs[i] = log_probs[i] + sampleFromGumbel(rng);
  }
}

// Returns the index of the top token.
template <typename T>
static int32_t argmax(const std::span<T> probs) {
  if (probs.empty()) {
    return -1;
  }

  const auto result = std::max_element(probs.begin(), probs.end());
  const size_t id   = std::distance(probs.begin(), result);

  return static_cast<int32_t>(id);
}

// Return the top-k indices of the input span using a min-heap
template <typename T>
static std::vector<int32_t> topK(const std::span<T> probs, int32_t k) {
  // Handle edge-case. This is usually an error call.
  if (k <= 0) {
    return {};
  }

  // If k is greater than number of elements, this is a pure sorting operation
  // This should also be fairly uncommon, and not the expected scenario for topK
  if (k >= probs.size()) {
    std::vector<int32_t> indices(probs.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&probs](int32_t a, int32_t b) {
      return probs[a] > probs[b];
    });
    return indices;
  }

  // Define min-heap elements as a pair of (probability, index)
  using ProbIdxPair = std::pair<T, int32_t>;
  std::priority_queue<ProbIdxPair, std::vector<ProbIdxPair>, std::greater<ProbIdxPair>> min_heap;

  for (size_t i = 0; i < probs.size(); i++) {
    if (min_heap.size() < k) {
      min_heap.push({probs[i], i});
    } else if (probs[i] > min_heap.top().first) {
      min_heap.pop();
      min_heap.push({probs[i], i});
    }
  }

  // Extract indices from the min-heap
  std::vector<int32_t> indices(k);
  size_t k_idx = k - 1;
  while (!min_heap.empty()) {
    indices[k_idx--] = min_heap.top().second;
    min_heap.pop();
  }

  return indices;
}

template <typename T>
static inline std::vector<int32_t> topK(Tensor& logits, int32_t k) {
  return topK<T>(std::span(reinterpret_cast<T*>(logits.getData()), logits.getSize()), k);
}

// Get top-p elements from the input vector (vec)
// note:
//  -  element in the vec is a pair of index and probability
//  -  vec will be inplace modified
//  -  finally the first n_remain elements in vec will be the top-p elements, but UNSORTED
//  -  n_remain elements will be returned
//  -  first_try_pos can be used to speedup processing heuristically,
//     set to -1 to disable heuristical speedup
template <typename T>
size_t partitionTopP(std::vector<std::pair<int32_t, T>>& vec,
                     float top_p,
                     int32_t first_try_pos,
                     size_t min_keep = 1) {
  static_assert(std::is_floating_point<T>::value);
  size_t i_start = 0;
  size_t i_end   = vec.size();  // not included

  T accum_left_sum = 0.0;  // sum of vec[:i_start]  (i_start is not included)
  T subarray_sum   = 1;    // sum of vec[i_start:i_end]  (i_end is not included)
  auto greater     = [](const std::pair<int32_t, T>& a, const std::pair<int32_t, T>& b) {
    return a.second > b.second;
  };

  size_t size                  = i_end - i_start;
  size_t m                     = size >> 1;
  size_t closest_partition_pos = vec.size();  // the closest partition position to min_keep,
                                              // but larger than min_keep

  if (first_try_pos > 0) {
    m = std::min(static_cast<size_t>(first_try_pos), size);
  }

  while (i_start < i_end) {
    if (m == 0) {
      break;
    }

    if (i_start + m > min_keep) {
      closest_partition_pos = std::min(closest_partition_pos, i_start + m);
    }

    // partition vec[i_start:i_end] at the position "i_start+m"
    // [* * * * * * * * * * * * * * * * * * * * * * * * *]
    //      ^                 ^                     ^
    //     i_start        i_start + m             i_end (not included)

    // which satisfies:
    //    - any element in vec[i_start:i_start+m] is larger  than vec[i_start+m]
    //    - any element in vec[i_start+m+1:i_end] is smaller than vec[i_start+m]
    std::nth_element(
        vec.begin() + i_start, vec.begin() + i_start + m, vec.begin() + i_end, greater);

    // calculate the sum of vec[i_start:i_start + m]  (i_start + m is not included)
    T subarray_left_sum = 0;
    for (size_t i = i_start; i < i_start + m; i++) {
      subarray_left_sum += vec[i].second;
    }

    if (subarray_left_sum + accum_left_sum < top_p) {
      // do next iter on right sub array [i_start+m:i_end]
      i_start = i_start + m;
      m       = (i_end - i_start) >> 1;
      accum_left_sum += subarray_left_sum;
      subarray_sum = subarray_sum - subarray_left_sum;
    } else {
      // do next iter on left sub array [i_start:i_start+m]
      i_end = i_start + m;
      m     = (i_end - i_start) >> 1;
    }
  }

  size_t n_remain = i_start + 1;
  if (n_remain < min_keep) {
    std::nth_element(
        vec.begin() + n_remain, vec.begin() + min_keep, vec.begin() + closest_partition_pos);
    n_remain = min_keep;
  }
  return n_remain;
}

// Utility function to penalize logits, if penalize limits are set
template <typename T>
void applyPenalty(Tensor logitsTensor, const Penalty& penalty, int32_t streamIdx = 0) {
  if (penalty.m_tokenFreqMap.size() == 0) return;

  std::span<T> logits =
      std::span(reinterpret_cast<T*>(logitsTensor.getData()), logitsTensor.getSize());

  for (const auto& [tokenIdx, tokenFreq] : penalty.m_tokenFreqMap[streamIdx]) {
    QUALLA_ASSERT(tokenIdx >= 0);
    QUALLA_ASSERT(tokenFreq > 0);

    TensorQuantizationParams qp = logitsTensor.getQuantizationParams();
    const double scale          = qp.scale;
    const int32_t offset        = qp.offset;
    float logitFloatVal         = (static_cast<float>(logits[tokenIdx]) + offset) * scale;

    // penalize for repetition
    if (logitFloatVal <= 0) {
      logitFloatVal *= penalty.m_penaltyRepeat;
    } else {
      logitFloatVal /= penalty.m_penaltyRepeat;
    }

    // penalize for presence and freq
    logitFloatVal -= static_cast<float>(tokenFreq) * penalty.m_penaltyFreq +
                     static_cast<float>(tokenFreq > 0) * penalty.m_penaltyPresent;

    //  update the logits value.
    logits[tokenIdx] = static_cast<T>(logitFloatVal / scale - offset);
  }
}

template <typename T>
struct IndexedQuantLogits : genie::profiling::Traceable {
  std::mt19937& rng;
  Tensor logitsTensor;
  std::span<T> logits;
  std::vector<float> probs;
  std::vector<int32_t> indices;
  Penalty& m_penalty;
  bool probs_valid;
  bool sorted;

  IndexedQuantLogits(std::shared_ptr<genie::profiling::TraceLogger> traceLogger,
                     Tensor logitsTensor,
                     std::mt19937& r,
                     Penalty& penalty)
      : Traceable(traceLogger),
        rng(r),
        logitsTensor(logitsTensor),
        indices(logitsTensor.getSize()),
        m_penalty(penalty),
        probs_valid(false),
        sorted(false) {
    GENIE_TRACE();
    // Technical note
    // The probs and indices vectors here may not be used/needed in many cases.
    // However, they are both large vectors (~vocab_size) and take up valuable runtime
    // Optimize the hot-path by changing to lazy initialization should avoid runtime overheads
    std::iota(indices.begin(), indices.end(), 0);
    logits = std::span(reinterpret_cast<T*>(logitsTensor.getData()), logitsTensor.getSize());
  }

  const char* getTraceNamespace() const override { return "IndexedLogits"; }

  size_t size(void) const { return logits.size(); }

  // Performs a partial sort or a full sort depending on k.
  size_t sort(size_t k = 0) {
    GENIE_TRACE();
    size_t logits_size = logits.size();

    k = k == 0 ? logits_size : k;
    k = std::min(k, logits_size);

    // Logits have already been fully sorted, so no computation required
    if (sorted && k <= logits_size) {
      logits = logits.first(k);
      indices.resize(k);
      if (probs_valid) {
        probs.resize(k);
      }
      return k;
    }

    // Partially-sort the topK indices based on the logits values
    std::partial_sort(
        indices.begin(), indices.begin() + k, indices.end(), [this](int32_t a, int32_t b) {
          return logits[a] > logits[b];
        });
    indices.resize(k);

    // FIXME: avoid overwriting input logits (Fixed?)
    if (probs_valid) {
      std::vector<T> tmp(k);
      std::vector<float> tmpf(k);
      for (size_t i = 0; i < k; i++) {
        tmp[i]  = logits[indices[i]];
        tmpf[i] = probs[indices[i]];
      }
      memcpy(const_cast<T*>(logits.data()), tmp.data(), k * sizeof(T));
      memcpy(probs.data(), tmpf.data(), k * sizeof(float));
      probs.resize(k);
    } else {
      std::vector<T> tmp(k);
      for (size_t i = 0; i < k; i++) {
        tmp[i] = logits[indices[i]];
      }
      memcpy(const_cast<T*>(logits.data()), tmp.data(), k * sizeof(T));
    }

    logits = logits.first(k);
    sorted = true;
    return k;
  }

  /**
   * Calculates softmax across topN probabilities and return topK
   */
  void softmaxTopK(float temp = 1.f, size_t k = 0, size_t n = 0) {
    GENIE_TRACE();
    // TODO: This may be identical to performing topK(n) -> softmax(temp) -> topK(k)
    QUALLA_ASSERT(temp > 0.f);
    QUALLA_ASSERT(k <= n);

    size_t logits_size = logits.size();

    // Compute n - number of indices to use for softmax
    n = this->sort(n);

    // Calculate k - number of indices to select for topK
    k = k == 0 ? logits_size : k;
    k = std::min(k, logits_size);
    k = std::min(k, n);

    indices.resize(k);  // Filter the topK indices

    const T max_logit = logits[0];  // Logits have already been sorted by sort()

    TensorQuantizationParams qp = logitsTensor.getQuantizationParams();
    auto scale                  = qp.scale;
    auto offset                 = qp.offset;
    float max_logit_float       = (static_cast<float>(max_logit) + offset) * scale;

    float max_scaled = max_logit_float / temp;
    float sum_exp    = 0.0f;

    auto multFactor  = scale / temp;
    auto additionVal = ((scale * offset) / temp) - max_scaled;

    probs.resize(n);  // Make sure probs has enough space for all n probabilities
#if !defined(__x86_64__)
    PRAGMA_LOOP_VECTORIZE
#endif  // __x86_64__
    // Calculate softmax over the topN probabilities
    for (size_t i = 0; i < n; i++) {
      const float p = std::exp((static_cast<float>(logits[i]) * multFactor) + additionVal);
      probs[i]      = p;
      sum_exp += p;
    }

    // Calculate topK probabilities to return
    PRAGMA_LOOP_VECTORIZE
    for (size_t i = 0; i < k; i++) {
      probs[i] /= sum_exp;
    }

    logits = logits.first(k);
    probs.resize(k);
    probs_valid = true;
  }

  // Does softmax in-place given a set of logits and a scaling temperature.
  void softmax(float temp = 1.f) {
    GENIE_TRACE();
    QUALLA_ASSERT(temp > 0.f);

    T max_logit;

    if (sorted) {
      max_logit = logits[0];
    } else {
      auto max_iter = std::max_element(logits.begin(), logits.end());
      max_logit     = *max_iter;
    }

    TensorQuantizationParams qp = logitsTensor.getQuantizationParams();
    auto scale                  = qp.scale;
    auto offset                 = qp.offset;
    float max_logit_float       = (static_cast<float>(max_logit) + offset) * scale;

    float max_scaled = max_logit_float / temp;
    float sum_exp    = 0.0f;

    auto multFactor  = scale / temp;
    auto additionVal = ((scale * offset) / temp) - max_scaled;

    const size_t num_logits = logits.size();
    probs.resize(num_logits);  // Make sure probs has enough space for all probabilities

#if !defined(__x86_64__)
    PRAGMA_LOOP_VECTORIZE
#endif  // __x86_64__
    for (size_t i = 0; i < num_logits; i++) {
      float p  = std::exp((static_cast<float>(logits[i]) * multFactor) + additionVal);
      probs[i] = p;
      sum_exp += p;
    }

    PRAGMA_LOOP_VECTORIZE
    for (size_t i = 0; i < num_logits; i++) {
      probs[i] /= sum_exp;
    }

    probs_valid = true;
  }

  void logSoftmax(float temp = 1.f) {
    GENIE_TRACE();
    QUALLA_ASSERT(temp > 0.f);
    float max_logit;

    if (sorted) {
      max_logit = logits[0];
    } else {
      auto max_iter = std::max_element(logits.begin(), logits.end());
      max_logit     = *max_iter;
    }

    TensorQuantizationParams qp = logitsTensor.getQuantizationParams();
    auto scale                  = qp.scale;
    auto offset                 = qp.offset;
    float max_logit_float       = (static_cast<float>(max_logit) + offset) * scale;

    // log(e^x / sum(e^x)) -> log(e^x) - log(sum(e^x))
    // We're still using the probs vector, despite the outputs technically
    // being log probabilities.

    float max_scaled = max_logit_float / temp;
    float sum_exp    = 0.0f;

    auto multFactor  = scale / temp;
    auto additionVal = ((scale * offset) / temp) - max_scaled;

    const size_t num_logits = logits.size();
    probs.resize(num_logits);  // Make sure probs has enough space for all probabilities

#if !defined(__x86_64__)
    PRAGMA_LOOP_VECTORIZE
#endif  // __x86_64__
    for (size_t i = 0; i < num_logits; i++) {
      float p  = (static_cast<float>(logits[i]) * multFactor) + additionVal;
      probs[i] = p;
      sum_exp += std::exp(p);
    }

    float log_sum_exp = std::log(sum_exp);
    PRAGMA_LOOP_VECTORIZE
    for (size_t i = 0; i < num_logits; i++) {
      probs[i] -= log_sum_exp;
    }

    probs_valid = true;
  }

  // Performs top-k
  void topK(int32_t k) {
    GENIE_TRACE();
    QUALLA_ASSERT(k > 0);
    k = this->sort(k);  // Sorts (if necessary) and truncates indices/probs/logits to size k
  }

  // Performs top-p in-place.
  // Note: the remained logits/probs are UNSORTED
  void topP(float p, size_t min_keep = 1) {
    GENIE_TRACE();
    if (p >= 1) return;

    if (!probs_valid) this->softmax();

    if (sorted) {
      // The probs are sorted, so find top-p elements directly

      // Compute the cumulative probabilities
      float cum_sum    = 0.0;
      size_t last_idx  = logits.size() - 1;
      size_t n_to_trim = 0;

      for (size_t i = last_idx; i > 0; --i) {
        cum_sum += probs[i];
        if (cum_sum <= 1.0 - p) {
          n_to_trim++;
        } else {
          break;
        }
      }

      size_t n_remain = logits.size() - n_to_trim;
      if (n_remain < min_keep) {
        n_remain += min_keep - n_remain;
      }

      logits = logits.first(n_remain);
      probs.resize(n_remain);
      indices.resize(n_remain);

    } else {
      // The probs are not sorted, so using binary partition to find top-p elements,
      // which is much faster than sorting for large vocab size

      // pack index/logit into one array, to improve data locality
      const size_t num_logits = logits.size();
      std::vector<std::pair<int32_t, float>> elements(num_logits);
      PRAGMA_LOOP_VECTORIZE
      for (size_t i = 0; i < num_logits; ++i) {
        elements[i] = std::make_pair(indices[i], probs[i]);
      }

      // normally probs are only concentrated in 1-100 labels
      // so try the first parition with a not very large value (like 4096)
      // will speedup the topp_by_nth in most of the cases
      int first_try_pos = TOPP_SAMPLER_INITIAL_PARTITION_POINT;

      // however, if the logits size is small, we don't need to this heuristic acceleration
      if (logits.size() < first_try_pos * 2) first_try_pos = -1;
      size_t n_remain = partitionTopP(elements, p, first_try_pos, min_keep);

      indices.resize(n_remain);
      probs.resize(n_remain);

      PRAGMA_LOOP_VECTORIZE
      for (size_t i = 0; i < n_remain; i++) {
        indices[i] = elements[i].first;
        probs[i]   = elements[i].second;
      }

      std::vector<T> temp_logits(n_remain);
      std::generate(temp_logits.begin(), temp_logits.end(), [this, elements]() {
        static uint32_t index = 0;
        return logits[elements[index].first];
      });

      memcpy(const_cast<T*>(logits.data()), temp_logits.data(), n_remain * sizeof(T));
      logits = logits.first(n_remain);
    }
  }

  // Greedy sampling
  int32_t sampleGreedyUnsorted(bool skipProbs = false) {
    GENIE_TRACE();
    auto result = std::max_element(logits.begin(), logits.end());
    size_t id   = std::distance(logits.begin(), result);

    if (!skipProbs) {
      probs.clear();
      probs.resize(logits.size(), 0.0f);
      probs[id]   = 1.0;
      probs_valid = true;
    }

    return int32_t(id);
  }

  // Sampling from prob distribution
  int32_t sampleFromProbs() {
    GENIE_TRACE();
    QUALLA_ASSERT(probs_valid);
    int32_t idx = qualla::sampleFromProbs<float>(std::span{probs.data(), probs.size()}, rng);
    return int32_t(indices[idx]);
  }

  // Sampling with Gumbel Max
  int32_t sampleUsingGumbelMax() {
    GENIE_TRACE();
    QUALLA_ASSERT(probs_valid);
    // probs here must be log-probabilities
    int32_t idx = qualla::sampleUsingGumbelMax<float>(std::span{probs.data(), probs.size()}, rng);
    return int32_t(indices[idx]);
  }

  // add gumbel noise to the logits
  bool addGumbelNoise() {
    // probs here must be log-probabilities
    qualla::addGumbelNoise<float>(probs, rng);
    return true;
  }

  // penalize the logits, if penalize limits are set
  void penalizeLogits(int32_t streamIdx = 0) {
    GENIE_TRACE();
    applyPenalty<T>(logitsTensor, m_penalty, streamIdx);
  }
};

}  // namespace qualla

#endif  // QUALLA_DETAIL_SAMPLER_UTILS_HPP
