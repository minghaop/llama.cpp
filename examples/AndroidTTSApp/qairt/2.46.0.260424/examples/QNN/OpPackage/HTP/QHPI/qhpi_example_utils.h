// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.

#ifndef QHPI_EXAMPLE_UTILS_H
#define QHPI_EXAMPLE_UTILS_H 1

#include <hexagon_types.h>

#define TILEDATASETUP(adrtab, nextTabCol, nextTabRow, h, w, d)                 \
  {                                                                            \
    .addr = (uint8_t **)(adrtab),                                              \
    .offsetTCol = static_cast<uint32_t>(nextTabCol),                           \
    .offsetTRow = static_cast<uint32_t>(nextTabRow),                           \
    .width = static_cast<uint32_t>(w), .height = static_cast<uint32_t>(h),     \
    .depth = static_cast<uint32_t>(d)                                          \
  }

#define STRINGIZE_DETAIL(X) #X
#define STRINGIZE(X) STRINGIZE_DETAIL(X)
#define THIS_PKG_NAME_STR STRINGIZE(THIS_PKG_NAME)

/////////////////////////////////////////
// 'aligned_buffer' classes
// On hexagon we can make the compiler align
// it by putting an HVX vector in the union;
// on x86 it's done manually
/////////////////////////////////////////
template <unsigned NVECS> struct aligned_buffer_base {
  static_assert(NVECS >= 1);

protected:
#ifdef __hexagon__
  static constexpr bool manual_align = false;
#else
  static constexpr bool manual_align = true;
#endif
  union {
    uint32_t u32arr[NVECS * 32 + (manual_align ? 31 : 0)];
#ifdef __hexagon__
    HVX_Vector varr[NVECS];
#endif
  } u;
  API_EXPORT void *arr_addr() const {
    if constexpr (manual_align) {
      size_t tmp = size_t(&(u.u32arr[0]));
      tmp = (tmp + 127) & ~size_t(127);
      return (void *)tmp;
    } else {
      return (void *)&(u.u32arr[0]);
    }
  }
};

//
// 'arrays' of NBUFS tile buffers...
//  call 'buf(i)' method, with  i in range 0..NBUFS-1, to get a pointer to one
//  of the buffers.
//
template <unsigned NBUFS, unsigned NVECS>
struct tile_buffers_template : public aligned_buffer_base<NBUFS * NVECS> {
  using Parent = aligned_buffer_base<NBUFS * NVECS>;

public:
#ifdef SAFE_ALLOC
  // For safety, clear everything to 0 so that if we load less then the size
  // of a tile, memory will have a deterministic value.
  tile_buffers_template() {
    // Clear memory if compiled with the SAFE_ALLOC option.
    memset(Parent::u32arr, 0, sizeof(Parent::u32arr));
  }
#endif

  API_EXPORT uint8_t *buf(unsigned i = 0) {
    return reinterpret_cast<uint8_t *>(this->arr_addr()) + NVECS * 128 * i;
  };
  API_EXPORT uint8_t const *buf(unsigned i = 0) const {
    return reinterpret_cast<uint8_t const *>(this->arr_addr()) +
           NVECS * 128 * i;
  };
};

struct TileData {
  uint8_t **addr;
  uint32_t offsetTCol;
  uint32_t offsetTRow;
  uint32_t width;
  uint32_t height;
  uint32_t depth;
};

/**
 * Accessor for quantized tensor elements with automatic float conversion.
 *
 * Transparently converts between quantized storage (type Elt) and float values
 * using the formula: quantized = round(float / stepsize + zero_offset)
 */
template <typename Elt> struct QElementRef {
  QHPI_Quant_Parameters params;
  Elt *location;
  Elt operator=(float value) {
    if constexpr (std::is_same_v<Elt, float>) {
      *location = value;
    } else {
      *location = static_cast<Elt>(
          std::round(value / params.stepsize + params.zero_offset));
    }
    return *location;
  }
  operator float() {
    float value = *location;
    return (value - params.zero_offset) * params.stepsize;
  }
};

/**
 * Accessor for 4D crouton-layout quantized tensors with block-tiled memory.
 *
 * Maps logical 4D indices (batch, height, width, depth) to physical block-tiled
 * layout with blocks of size [8 × WIDTH × 32] where WIDTH = 8/sizeof(Elt)
 *
 * Provides both raw element access and automatic
 * quantization/dequantization via operator().
 */
template <typename Elt> struct Crouton4 {
  static inline uint32_t floor(uint32_t x, uint32_t chunk) { return x / chunk; }
  static inline uint32_t ceil(uint32_t x, uint32_t chunk) {
    return (x + chunk - 1) / chunk;
  }
  static constexpr uint32_t WIDTH = 8 / sizeof(Elt);
  static constexpr size_t element_size = sizeof(Elt);
  uint32_t multipliers[3];
  QHPI_Quant_Parameters params;
  QHPI_Shape shape;
  Elt **block_table;
  uint32_t block_table_length;
  Crouton4(const QHPI_Tensor *tensor) {
    shape = qhpi_tensor_shape(tensor);
    block_table = reinterpret_cast<Elt **>(qhpi_tensor_block_table(tensor));
    block_table_length = qhpi_tensor_block_table_length(tensor);
    multipliers[2] = ceil(shape.dims[3], 32);
    multipliers[1] = multipliers[2] * ceil(shape.dims[2], WIDTH);
    multipliers[0] = multipliers[1] * ceil(shape.dims[1], 8);
    params = qhpi_tensor_quant_parameters(tensor);
  }
  std::pair<uint32_t, uint32_t> split(uint32_t b, uint32_t h, uint32_t w,
                                      uint32_t d) const {
    uint32_t block = b * multipliers[0] + floor(h, 8) * multipliers[1] +
                     floor(w, WIDTH) * multipliers[2] + floor(d, 32);
    uint32_t offset = 0 + (h % 8) * (WIDTH * 32) + (w % WIDTH) * 32 + d % 32;
    return {block, offset};
  }
  Elt &get_raw(uint32_t b, uint32_t h, uint32_t w, uint32_t d) {
    auto [block, offset] = split(b, h, w, d);
    return block_table[block][offset];
  }
  Elt &get_raw(uint32_t b, uint32_t h, uint32_t w, uint32_t d) const {
    auto [block, offset] = split(b, h, w, d);
    return block_table[block][offset];
  }
  Elt *get_raw_addr(uint32_t b, uint32_t h, uint32_t w, uint32_t d) {
    auto [block, offset] = split(b, h, w, d);
    return &block_table[block][offset];
  }
  Elt *block_ptr(uint32_t b, uint32_t h, uint32_t w, uint32_t d) {
    auto [block, offset] = split(b, h, w, d);
    return block_table[block];
  }
  float operator()(uint32_t b, uint32_t h, uint32_t w, uint32_t d) const {
    auto [block, offset] = split(b, h, w, d);
    return QElementRef<Elt>{params, &block_table[block][offset]};
  }
  QElementRef<Elt> operator()(uint32_t b, uint32_t h, uint32_t w, uint32_t d) {
    return QElementRef<Elt>{params, &get_raw(b, h, w, d)};
  }
};

template <typename Elt> struct Flat4 {
  uint32_t multipliers[4];
  Elt *data;
  QHPI_Quant_Parameters params;
  QHPI_Shape shape;
  Flat4(const QHPI_Tensor *tensor) {
    shape = qhpi_tensor_shape(tensor);
    data = reinterpret_cast<Elt *>(qhpi_tensor_raw_data(tensor));
    multipliers[2] = shape.dims[3];
    multipliers[1] = multipliers[2] * shape.dims[2];
    multipliers[0] = multipliers[1] * shape.dims[1];
    params = qhpi_tensor_quant_parameters(tensor);
  }
  Elt &get_raw(uint32_t b, uint32_t h, uint32_t w, uint32_t d) {
    uint32_t offset =
        b * multipliers[0] + h * multipliers[1] + w * multipliers[2] + d;
    return data[offset];
  }
  Elt &get_raw(uint32_t b, uint32_t h, uint32_t w, uint32_t d) const {
    uint32_t offset =
        b * multipliers[0] + h * multipliers[1] + w * multipliers[2] + d;
    return data[offset];
  }
  Elt *get_raw_addr(uint32_t b, uint32_t h, uint32_t w, uint32_t d) const {
    uint32_t offset =
        b * multipliers[0] + h * multipliers[1] + w * multipliers[2] + d;
    return &data[offset];
  }
  float operator()(uint32_t b, uint32_t h, uint32_t w, uint32_t d) const {
    return QElementRef<Elt>{params, &get_raw(b, h, w, d)};
  }
  QElementRef<Elt> operator()(uint32_t b, uint32_t h, uint32_t w, uint32_t d) {
    return QElementRef<Elt>{params, &get_raw(b, h, w, d)};
  }
};

inline QHPI_OpRef create_int_constant(const QHPI_Op *root, uint32_t width,
                                      uint32_t depth, void *data) {
  QHPI_OutputDef outDef = {QHPI_Int32,
                           {0, 0},
                           {
                               4,                    // rank
                               {1, 1, width, depth}, // dims
                           }};
  return qhpi_op_reference(
      qhpi_op_create_constant(root, &outDef, sizeof(uint32_t) * width * depth,
                              data),
      0);
}

inline QHPI_OpRef create_float_constant(const QHPI_Op *root, uint32_t width,
                                        uint32_t depth, void *data) {
  QHPI_OutputDef outDef = {QHPI_Float32,
                           {0, 0},
                           {
                               4,                    // rank
                               {1, 1, width, depth}, // dims
                           }};
  return qhpi_op_reference(
      qhpi_op_create_constant(root, &outDef, sizeof(uint32_t) * width * depth,
                              data),
      0);
}

#endif
