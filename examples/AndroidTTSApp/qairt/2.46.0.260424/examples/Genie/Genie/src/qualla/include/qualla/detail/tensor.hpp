//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <cstddef>
#include <cstring>
#include <vector>

namespace qualla {

enum TensorDataType {
  TENSOR_DATATYPE_UFIXED_POINT_8  = 0x01,
  TENSOR_DATATYPE_UFIXED_POINT_16 = 0x02,
  TENSOR_DATATYPE_INT_32          = 0x03,
  TENSOR_DATATYPE_INT_64          = 0x04,
  TENSOR_DATATYPE_FLOAT_POINT_16  = 0x05,
  TENSOR_DATATYPE_FLOAT_32        = 0x06,
  TENSOR_DATATYPE_UNKNOWN         = 0xFF
};

enum class LayerType {
  INPUT,
  OUTPUT,
  ATTN_MASK,
  ANCHOR,
  VALID_MASK,
  POS_SIN,
  POS_COS,
  POS_IDS,
  CACHE_INDEX,
  TOKEN_TYPE_IDS,
  POOL_OUTPUT,
  SEQ_OUTPUT,
  INPUT_EMBED,
  FULL_ATTN_MASK,
  WINDOW_ATTN_MASK,
  CROSS_ATTN_STATES,
  CROSS_ATTN_MASK,
  EAGLET_DM_HIDDEN_STATES_IN,
  EAGLET_DM_HIDDEN_STATES_OUT,
  EAGLET_TM_HIDDEN_STATES_IN,

  // model specific
  FULL_TEXT_ROW_MASK,  // Llama3.2-11B
  PRETILE_EMBED,       // Llama3.2-11B
  POSTTILE_EMBED,      // Llama3.2-11B
  GATED_POS_EMBED,     // Llama3.2-11B

  // Unet
  TIMESTEP
};

typedef struct {
  double scale;
  int32_t offset;
} TensorQuantizationParams;

/*
 *  This class has 2 uses: one where it solely points to logit data but doesn't own it,
 *  and one where it owns logit data in a vector of float and points to it using the same
 *  void* pointer.
 */
class Tensor {
 private:
  void* data                                  = nullptr;
  TensorDataType dataType                     = TENSOR_DATATYPE_UNKNOWN;
  TensorQuantizationParams quantizationParams = {1, 0};
  size_t numElements                          = 0;
  std::vector<uint32_t> dimensions;
  bool userOwned = true;

 public:
  std::vector<float> logits;

  // Default constructor
  Tensor() = default;

  // Constructor with parameters
  Tensor(void* data,
         TensorDataType dataType,
         TensorQuantizationParams quantizationParams,
         size_t numElements,
         const std::vector<uint32_t>& dimensions,
         bool user_owned)
      : dataType(dataType),
        quantizationParams(quantizationParams),
        numElements(numElements),
        dimensions(dimensions),
        userOwned(user_owned) {
    if (userOwned) {
      // User owns the data, just point to it
      this->data = data;
    } else {
      // We need to allocate memory based on dataType and numElements
      size_t byteWidth = getByteWidth(dataType);
      if (byteWidth > 0 && numElements > 0) {
        size_t totalBytes = numElements * byteWidth;
        this->data        = malloc(totalBytes);
        if (this->data && data) {
          // Copy data if source data is provided
          std::memcpy(this->data, data, totalBytes);
        } else if (this->data) {
          // Zero-initialize if no source data
          std::memset(this->data, 0, totalBytes);
        }
      }
    }
  }

  // Destructor to free allocated memory
  ~Tensor() {
    if (!userOwned && data) {
      free(data);
      data = nullptr;
    }
  }

  // Copy constructor
  Tensor(const Tensor& other)
      : dataType(other.dataType),
        quantizationParams(other.quantizationParams),
        numElements(other.numElements),
        dimensions(other.dimensions),
        userOwned(other.userOwned),
        logits(other.logits) {
    if (!userOwned && other.data) {
      size_t byteWidth = getByteWidth(dataType);
      if (byteWidth > 0 && numElements > 0) {
        size_t totalBytes = numElements * byteWidth;
        this->data        = malloc(totalBytes);
        if (this->data) {
          std::memcpy(this->data, other.data, totalBytes);
        }
      }
    } else {
      this->data = other.data;
    }
  }

  // Copy assignment operator
  Tensor& operator=(const Tensor& other) {
    if (this != &other) {
      // Free existing data if we own it
      if (!userOwned && data) {
        free(data);
        data = nullptr;
      }

      dataType           = other.dataType;
      quantizationParams = other.quantizationParams;
      numElements        = other.numElements;
      dimensions         = other.dimensions;
      userOwned          = other.userOwned;
      logits             = other.logits;

      if (!userOwned && other.data) {
        size_t byteWidth = getByteWidth(dataType);
        if (byteWidth > 0 && numElements > 0) {
          size_t totalBytes = numElements * byteWidth;
          this->data        = malloc(totalBytes);
          if (this->data) {
            std::memcpy(this->data, other.data, totalBytes);
          }
        }
      } else {
        this->data = other.data;
      }
    }
    return *this;
  }

  void* getData() const { return this->data; }
  void setData(void* data) { this->data = data; }

  TensorDataType getDataType() const { return this->dataType; }
  void setDataType(TensorDataType dataType) { this->dataType = dataType; }

  size_t getSize() const { return this->numElements; }
  void setSize(size_t numElements) { this->numElements = numElements; }

  std::vector<uint32_t> getDimensions() const { return this->dimensions; }
  void setDimensions(const std::vector<uint32_t>& dims) { this->dimensions = dims; }

  TensorQuantizationParams getQuantizationParams() const { return this->quantizationParams; }
  void setQuantizationParams(double scale, int32_t offset) {
    this->quantizationParams.scale  = scale;
    this->quantizationParams.offset = offset;
  }

  Tensor getIndexedTensor(size_t index, size_t vocab, bool dynamicExtent = false) const {
    size_t bytewidth = getByteWidth(this->getDataType());
    if (bytewidth == 0) {
      // Invalid data type - return empty tensor
      return Tensor();
    }

    Tensor toReturn;

    auto returnData = (uint8_t*)(this->getData());
    returnData += index * vocab * bytewidth;
    toReturn.setData((void*)returnData);

    toReturn.setDataType(this->getDataType());

    if (!dynamicExtent)
      toReturn.setSize(vocab);
    else
      toReturn.setSize((this->numElements) - (index * vocab));

    toReturn.setQuantizationParams(this->quantizationParams.scale, this->quantizationParams.offset);
    return toReturn;
  }

  static Tensor deepCopy(const Tensor& src) {
    Tensor copy(src);
    if (src.userOwned) {
      copy.userOwned = false;
      // We need to allocate memory based on dataType and numElements
      size_t byteWidth = Tensor::getByteWidth(src.dataType);
      if (byteWidth > 0 && src.numElements > 0) {
        size_t totalBytes = src.numElements * byteWidth;
        copy.data         = malloc(totalBytes);
        if (copy.data && src.data) {
          // Copy data if source data is provided
          std::memcpy(copy.data, src.data, totalBytes);
        } else if (copy.data) {
          // Zero-initialize if no source data
          std::memset(copy.data, 0, totalBytes);
        }
      }
    }
    return copy;
  }

  // Helper function to get byte width from data type
  static size_t getByteWidth(TensorDataType dataType) {
    switch (dataType) {
      case TENSOR_DATATYPE_UFIXED_POINT_8:
        return 1;
      case TENSOR_DATATYPE_UFIXED_POINT_16:
      case TENSOR_DATATYPE_FLOAT_POINT_16:
        return 2;
      case TENSOR_DATATYPE_FLOAT_32:
      case TENSOR_DATATYPE_INT_32:
        return 4;
      case TENSOR_DATATYPE_INT_64:
        return 8;
      default:
        return 0;
    }
  }
};

}  // namespace qualla
