//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <limits>
#include <unordered_map>

#include "read-gguf.hpp"

#if defined(__clang__)
#define DIAGNOSTIC_PUSH _Pragma("clang diagnostic push")
#define DIAGNOSTIC_POP  _Pragma("clang diagnostic pop")
#define DIAGNOSTIC_IGNORE _Pragma("clang diagnostic ignored \"-Wformat-nonliteral\"")
#define ATTRIBUTE_FORMAT __attribute__((format(printf, 1, 2)))
#elif defined(__GNUC__)
#define DIAGNOSTIC_PUSH _Pragma("GCC diagnostic push")
#define DIAGNOSTIC_POP  _Pragma("GCC diagnostic pop")
#define DIAGNOSTIC_IGNORE _Pragma("GCC diagnostic ignored \"-Wformat-nonliteral\"")
#define ATTRIBUTE_FORMAT __attribute__((format(printf, 1, 2)))
#else
#define DIAGNOSTIC_PUSH
#define DIAGNOSTIC_POP
#define DIAGNOSTIC_IGNORE
#define ATTRIBUTE_FORMAT
#endif

#define GGUF_CHECK_ERROR_NE(cmd, error)   \
  do {                                    \
    int x = cmd;                          \
    if (x != static_cast<int>((error))) { \
      goto exit;                          \
    }                                     \
  } while (0)

#define GGUF_CHECK_ERROR_EQ(cmd, error)   \
  do {                                    \
    int x = cmd;                          \
    if (x == static_cast<int>((error))) { \
      goto exit;                          \
    }                                     \
  } while (0)

#define GGUF_KEY_DECODER "cross_attention_decoder"

namespace gguf{

DIAGNOSTIC_PUSH
DIAGNOSTIC_IGNORE
ATTRIBUTE_FORMAT
char* stringFormatter(const char* format, ...) {
  va_list args;
  uint32_t length;

  va_start(args, format);
  length = static_cast<uint32_t>(vsnprintf(NULL, 0, format, args) + 1);
  va_end(args);

  if (static_cast<int32_t>(length) < 0) {
    return NULL;
  }

  char* string = static_cast<char*>(malloc(length * sizeof(char)));

  if (!string) {
    return NULL;
  }

  va_start(args, format);
  vsnprintf(string, length, format, args);
  va_end(args);

  return string;
}
DIAGNOSTIC_POP

enum class GGUFKeyType : int {
  // General
  GENERAL_ARCHITECTURE,
  GENERAL_QUANTIZATION_VERSION,
  GENERAL_ALIGNMENT,
  GENERAL_NAME,
  GENERAL_TOKENIZER,
  GENERAL_SOURCE_HF_REPO,
  GENERAL_FILE_TYPE,
  GENERAL_OUTPUT,
  // LLM Specific
  VOCAB_SIZE,
  CONNECTOR,
  ARCH_GATE,
  CONTEXT_LENGTH,
  EMBEDDING_LENGTH,
  EMBEDDING_PER_HEAD,
  BLOCK_COUNT,
  FEED_FORWARD_LENGTH,
  // Operation Specific
  OPERATION_NORMALIZATION,
  OPERATION_ACTIVATION,
  OPERATION_POSITIONAL_EMBEDDING,
  WPE_OFFSET,
  OPERATION_ROPE_COMPLEX_ORG,
  OPERATION_NORMALIZATION_EPS,
  OPERATION_ATTENTION_MODE,
  ROPE_SCALING_FACTOR_SHORT,
  ROPE_SCALING_FACTOR_LONG,
  ROPE_FACTOR_ATTN,
  // Attention Specific
  ATTENTION_HEAD_COUNT,
  ATTENTION_HEAD_COUNT_KV,
  ATTENTION_LAYERNORM_EPS,
  // RoPE Specific
  ROPE_NUM_ROTATION,
  ROPE_FREQ_BASE,
  ROPE_SCALE_LINEAR,
  // Tokenizer Specific
  TOKENIZER_MODEL,
  TOKENIZER_LIST,
  TOKENIZER_SCORES,
  TOKENIZER_BOS_ID,
  TOKENIZER_EOS_ID,
  TOKENIZER_UNK_ID,
  TOKENIZER_SEP_ID,
  TOKENIZER_PAD_ID,
  TOKENIZER_CLS_ID,
  // LoRA Specific
  ALPHA_VALUE,
  RANK_VALUE,
};

static const std::unordered_map<GGUFKeyType, std::string>& getGGUFKeyMap() {
  static const std::unordered_map<GGUFKeyType, std::string> s_ggufKeyMap {
    {GGUFKeyType::GENERAL_ARCHITECTURE,           "general.architecture"},
    {GGUFKeyType::GENERAL_QUANTIZATION_VERSION,   "general.quantization_version"},
    {GGUFKeyType::GENERAL_ALIGNMENT,              "general.alignment"},
    {GGUFKeyType::GENERAL_NAME,                   "general.name"},
    {GGUFKeyType::GENERAL_TOKENIZER,              "general.tokenizer"},
    {GGUFKeyType::GENERAL_SOURCE_HF_REPO,         "model.general.hf_hub_model_id"},
    {GGUFKeyType::GENERAL_FILE_TYPE,              "general.file_type"},
    {GGUFKeyType::GENERAL_OUTPUT,                 "model.general.output"},
    {GGUFKeyType::VOCAB_SIZE,                     "model.size.vocabulary"},
    {GGUFKeyType::CONNECTOR,                      "model.architecture.connector"},
    {GGUFKeyType::ARCH_GATE,                      "model.architecture.gating"},
    {GGUFKeyType::CONTEXT_LENGTH,                 "%s.context_length"},
    {GGUFKeyType::EMBEDDING_LENGTH,               "%s.embedding_length"},
    {GGUFKeyType::EMBEDDING_PER_HEAD,             "%s.embedding_per_head"},
    {GGUFKeyType::BLOCK_COUNT,                    "%s.block_count"},
    {GGUFKeyType::FEED_FORWARD_LENGTH,            "%s.feed_forward_length"},
    {GGUFKeyType::OPERATION_NORMALIZATION,        "model.operation.normalization"},
    {GGUFKeyType::OPERATION_ACTIVATION,           "model.operation.activation"},
    {GGUFKeyType::OPERATION_POSITIONAL_EMBEDDING, "model.operation.positional_embedding"},
    {GGUFKeyType::WPE_OFFSET,                     "model.operation.wpe_offset"},
    {GGUFKeyType::OPERATION_ROPE_COMPLEX_ORG,     "model.operation.rope_complex_organization"},
    {GGUFKeyType::OPERATION_NORMALIZATION_EPS,    "model.operation.normalization_epsilon"},
    {GGUFKeyType::OPERATION_ATTENTION_MODE,       "model.operation.attention_mode"},
    {GGUFKeyType::ROPE_SCALING_FACTOR_SHORT,      "model.operation.rope.scaling.factor.short"},
    {GGUFKeyType::ROPE_SCALING_FACTOR_LONG,       "model.operation.rope.scaling.factor.long"},
    {GGUFKeyType::ROPE_FACTOR_ATTN,               "model.operation.rope.scaling.attn_factor"},
    {GGUFKeyType::ATTENTION_HEAD_COUNT,           "%s.attention.head_count"},
    {GGUFKeyType::ATTENTION_HEAD_COUNT_KV,        "%s.attention.head_count_kv"},
    {GGUFKeyType::ATTENTION_LAYERNORM_EPS,        "%s.attention.layer_norm_epsilon"},
    {GGUFKeyType::ROPE_NUM_ROTATION,              "%s.rope.dimension_count"},
    {GGUFKeyType::ROPE_FREQ_BASE,                 "%s.rope.freq_base"},
    {GGUFKeyType::ROPE_SCALE_LINEAR,              "%s.rope.scale_linear"},
    {GGUFKeyType::TOKENIZER_MODEL,                "tokenizer.ggml.model"},
    {GGUFKeyType::TOKENIZER_LIST,                 "tokenizer.ggml.tokens"},
    {GGUFKeyType::TOKENIZER_SCORES,               "tokenizer.ggml.scores"},
    {GGUFKeyType::TOKENIZER_BOS_ID,               "tokenizer.bos_token_id"},
    {GGUFKeyType::TOKENIZER_EOS_ID,               "tokenizer.eos_token_id"},
    {GGUFKeyType::TOKENIZER_UNK_ID,               "tokenizer.unk_token_id"},
    {GGUFKeyType::TOKENIZER_SEP_ID,               "tokenizer.sep_token_id"},
    {GGUFKeyType::TOKENIZER_PAD_ID,               "tokenizer.pad_token_id"},
    {GGUFKeyType::TOKENIZER_CLS_ID,               "tokenizer.cls_token_id"},
    {GGUFKeyType::ALPHA_VALUE,                    "model.lora.alpha"},
    {GGUFKeyType::RANK_VALUE,                     "model.lora.rank"}
  };

  return s_ggufKeyMap;
}

enum class GGUFValueType : int {
  UINT8   = 0,
  INT8    = 1,
  UINT16  = 2,
  INT16   = 3,
  UINT32  = 4,
  INT32   = 5,
  FLOAT32 = 6,
  BOOL    = 7,
  STRING  = 8,
  ARRAY   = 9,
  UINT64  = 10,
  INT64   = 11,
  FLOAT64 = 12,
};

size_t getGGUFValueTypeSize (GGUFValueType type) {
  static const std::unordered_map<GGUFValueType, size_t> s_ggufValueTypeToSize {
    {GGUFValueType::UINT8,    sizeof(uint8_t)},
    {GGUFValueType::INT8,     sizeof(int8_t)},
    {GGUFValueType::UINT16,   sizeof(uint16_t)},
    {GGUFValueType::INT16,    sizeof(int16_t)},
    {GGUFValueType::UINT32,   sizeof(uint32_t)},
    {GGUFValueType::INT32,    sizeof(int32_t)},
    {GGUFValueType::FLOAT32,  sizeof(float)},
    {GGUFValueType::UINT64,   sizeof(uint64_t)},
    {GGUFValueType::INT64,    sizeof(int64_t)},
    {GGUFValueType::FLOAT64,  sizeof(double)},
    {GGUFValueType::BOOL,     sizeof(bool)},
    {GGUFValueType::STRING,   sizeof(char*)}
  };

  return s_ggufValueTypeToSize.contains(type) ? s_ggufValueTypeToSize.at(type)
                                              : std::numeric_limits<size_t>::max();
}

struct gguf_array {
  GGUFValueType type;
  uint64_t size;
  void* data;
};

union gguf_value {
  uint8_t uint8;
  int8_t int8;
  uint16_t uint16;
  int16_t int16;
  uint32_t uint32;
  int32_t int32;
  float float32;
  uint64_t uint64;
  int64_t int64;
  double float64;
  bool boolean;
  char* string;
  struct gguf_array array;
};

struct gguf_kv {
  char* key;
  GGUFValueType type;
  union gguf_value value;
};

struct gguf_tensor {
  static constexpr uint32_t MAX_DIM_SIZE = 4;

  char* name;
  uint32_t n_dim;
  uint64_t dim[MAX_DIM_SIZE];
  uint32_t type;
  uint64_t offset;
};

struct gguf_file {
  uint32_t magic;
  uint32_t version;
  uint64_t n_tensor;
  uint64_t n_kv;
  struct gguf_kv* kv;
  struct gguf_tensor* tensor_info;
};

void ggufFileFree(struct gguf_file* f) {
  if (!f) {
    return;
  }

  // Free key-value pairs
  for (size_t i = 0; i < f->n_kv; i++) {
    free(f->kv[i].key);
    if (f->kv[i].type == GGUFValueType::STRING) {
      free(f->kv[i].value.string);
    } else if (f->kv[i].type == GGUFValueType::ARRAY) {
      if (f->kv[i].value.array.type == GGUFValueType::STRING) {
        for (size_t j = 0; j < f->kv[i].value.array.size; j++) {
          free((static_cast<char**>(f->kv[i].value.array.data))[j]);
        }
      }
      free(f->kv[i].value.array.data);
    }
  }
  free(f->kv);

  // Free tensors
  for (size_t i = 0; i < f->n_tensor; i++) {
    free(f->tensor_info[i].name);
  }
  free(f->tensor_info);

  free(f);
}

static inline bool ggufStringRead(FILE* fp, char** string) {
  uint64_t length = 0;
  GGUF_CHECK_ERROR_NE(fread(&length, sizeof(uint64_t), 1, fp), 1);

  if (length <= std::numeric_limits<uint64_t>::max()) {
    *string = static_cast<char*>(malloc(length + 1));
    if (*string != nullptr) {
      GGUF_CHECK_ERROR_NE(fread(*string, sizeof(char), length, fp), length);
      (*string)[length] = '\0';
      return true;
    }
  }

exit:
  return false;
}

bool ggufFileRead(const char* file_name, struct gguf_file** file) {
  FILE* fp = fopen(file_name, "rb");
  if (!fp) {
    return false;
  }

  struct gguf_file* f = static_cast<struct gguf_file*>(calloc(1, sizeof(struct gguf_file)));
  if (!f) { goto exit; }

  // Read header
  GGUF_CHECK_ERROR_NE(fread(&f->magic, sizeof(uint32_t), 1, fp), 1);
  GGUF_CHECK_ERROR_NE(fread(&f->version, sizeof(uint32_t), 1, fp), 1);
  GGUF_CHECK_ERROR_NE(fread(&f->n_tensor, sizeof(uint64_t), 1, fp), 1);
  GGUF_CHECK_ERROR_NE(fread(&f->n_kv, sizeof(uint64_t), 1, fp), 1);

  // Parameter validation to prevent overalloc
  if (f->n_tensor > (std::numeric_limits<size_t>::max() / sizeof(*f->tensor_info))) { goto exit; }
  if (f->n_kv > (std::numeric_limits<size_t>::max() / sizeof(*f->kv))) { goto exit; }

  // Read key-value pairs
  f->kv = static_cast<struct gguf_kv*>(calloc(f->n_kv, sizeof(*f->kv)));
  if (!f->kv) { goto exit; }
  for (size_t i = 0; i < f->n_kv; i++) {
    struct gguf_kv* kv = &f->kv[i];

    // Read key
    GGUF_CHECK_ERROR_NE(ggufStringRead(fp, &kv->key), 1);

    // Read value type
    GGUF_CHECK_ERROR_NE(fread(&kv->type, sizeof(kv->type), 1, fp), 1);

    // Read value
    switch (kv->type) {
      case GGUFValueType::STRING:
        GGUF_CHECK_ERROR_NE(ggufStringRead(fp, &kv->value.string), 1);
        break;
      case GGUFValueType::ARRAY: {
        struct gguf_array* array = &kv->value.array;

        GGUF_CHECK_ERROR_NE(fread(&array->type, sizeof(array->type), 1, fp), 1);
        GGUF_CHECK_ERROR_NE(fread(&array->size, sizeof(array->size), 1, fp), 1);
        const size_t gguf_type_size = getGGUFValueTypeSize(array->type);
        if(array->size > (SIZE_MAX / gguf_type_size)) { goto exit; }

        array->data = calloc(array->size, gguf_type_size);
        if(!array->data) { goto exit; }

        if (array->type == GGUFValueType::STRING) {
          for (size_t j = 0; j < array->size; j++) {
            GGUF_CHECK_ERROR_NE(ggufStringRead(fp, &(static_cast<char**>(array->data))[j]), 1);
          }
        } else {
          GGUF_CHECK_ERROR_NE(fread(array->data, gguf_type_size, array->size, fp), array->size);
        }
        break;
      }
      case GGUFValueType::UINT8:
      case GGUFValueType::INT8:
      case GGUFValueType::UINT16:
      case GGUFValueType::INT16:
      case GGUFValueType::UINT32:
      case GGUFValueType::INT32:
      case GGUFValueType::FLOAT32:
      case GGUFValueType::BOOL:
      case GGUFValueType::UINT64:
      case GGUFValueType::INT64:
      case GGUFValueType::FLOAT64:
        GGUF_CHECK_ERROR_NE(fread(&kv->value, getGGUFValueTypeSize(kv->type), 1, fp), 1);
        break;
      default:
        goto exit;
    }
  }

  // Read tensor infos
  f->tensor_info = static_cast<struct gguf_tensor*>(calloc(f->n_tensor, sizeof(*f->tensor_info)));
  if (!f->tensor_info) { goto exit; }
  for (size_t i = 0; i < f->n_tensor; i++) {
    struct gguf_tensor* tensor = &f->tensor_info[i];

    // Read tensor name
    GGUF_CHECK_ERROR_NE(ggufStringRead(fp, &tensor->name), 1);

    // Read tensor rank
    GGUF_CHECK_ERROR_NE(fread(&tensor->n_dim, sizeof(tensor->n_dim), 1, fp), 1);
    if (tensor->n_dim > gguf_tensor::MAX_DIM_SIZE) {
      goto exit;
    }

    // Read tensor dims
    for (int64_t j = static_cast<int64_t>(tensor->n_dim) - 1; j >= 0; j--) {
      GGUF_CHECK_ERROR_NE(fread(&tensor->dim[j], sizeof(tensor->dim[j]), 1, fp), 1);
    }

    // Read tensor data type
    GGUF_CHECK_ERROR_NE(fread(&tensor->type, sizeof(tensor->type), 1, fp), 1);

    // Read tensor data offset
    GGUF_CHECK_ERROR_NE(fread(&tensor->offset, sizeof(tensor->offset), 1, fp), 1);
  }

  *file = f;
  fclose(fp);
  return true;

exit:
  ggufFileFree(f);
  fclose(fp);
  return false;
}

std::string ggufFilePrint(struct gguf_file *file) {
  std::streambuf* stdOutBuf = std::cout.rdbuf();
  std::ostringstream outStream;
  std::cout.rdbuf(outStream.rdbuf());

  char* magic = reinterpret_cast<char*>(&file->magic);
  std::cout << "magic         : " << std::string(magic, (magic + 4)) << std::endl;
  std::cout << "version       : " << file->version << std::endl;
  std::cout << "ti_data_count : " << file->n_tensor << std::endl;
  std::cout << "kv_data_count : " << file->n_kv << std::endl;

  for(size_t i = 0; i < file->n_kv; i++) {
    struct gguf_kv* kv = &file->kv[i];
    std::cout << "KEY :    " << std::setw(50) << kv->key;

    switch (kv->type) {
      case GGUFValueType::UINT8:
        std::cout << "\t VALUE : " << static_cast<int>(kv->value.uint8) << std::endl;
        break;
      case GGUFValueType::INT8:
        std::cout << "\t VALUE : " << static_cast<int>(kv->value.int8) << std::endl;
        break;
      case GGUFValueType::UINT16:
        std::cout << "\t VALUE : " << static_cast<int>(kv->value.uint16) << std::endl;
        break;
      case GGUFValueType::INT16:
        std::cout << "\t VALUE : " << static_cast<int>(kv->value.int16) << std::endl;
        break;
      case GGUFValueType::UINT32:
        std::cout << "\t VALUE : " << kv->value.uint32 << std::endl;
        break;
      case GGUFValueType::INT32:
        std::cout << "\t VALUE : " << kv->value.int32 << std::endl;
        break;
      case GGUFValueType::FLOAT32:
        std::cout << "\t VALUE : " << kv->value.float32 << std::endl;
        break;
      case GGUFValueType::UINT64:
        std::cout << "\t VALUE : " << kv->value.uint64 << std::endl;
        break;
      case GGUFValueType::INT64:
        std::cout << "\t VALUE : " << kv->value.int64 << std::endl;
        break;
      case GGUFValueType::FLOAT64:
        std::cout << "\t VALUE : " << kv->value.float64 << std::endl;
        break;
      case GGUFValueType::BOOL:
        std::cout << "\t VALUE : " << static_cast<int>(kv->value.boolean) << std::endl;
        break;
      case GGUFValueType::STRING:
        std::cout << "\t VALUE : " << kv->value.string << std::endl;
        break;
      case GGUFValueType::ARRAY:
        std::cout << "\t VALUE : ARR TYPE " << static_cast<int>(kv->value.array.type)
                  << " LENGTH " << kv->value.array.size << std::endl;
        break;
    }
  }

  for(size_t i = 0; i < file->n_tensor; i++) {
    struct gguf_tensor* tensor_info = &file->tensor_info[i];
    std::cout << "TENSOR : " << std::setw(50) << tensor_info->name;
    std::cout << "\t " << tensor_info->type;
    std::cout << "\t [ ";
    for(size_t j = 0; j < tensor_info->n_dim; j++) {
      std::cout << tensor_info->dim[j] << " ";
    }
    std::cout << "]";
    std::cout << "\t OFFSET : " << tensor_info->offset << std::endl;
  }
  std::cout << std::endl;

  std::cout.rdbuf(stdOutBuf);
  return outStream.str();
}

size_t ggufFindKey(struct gguf_file* file, const char* key) {
  if (file && key) {
    for (size_t i = 0; i < file->n_kv; i++) {
      struct gguf_kv* kv = &file->kv[i];
      if (!strcmp(key, kv->key)) {
        return i;
      }
    }
  }

  return std::numeric_limits<size_t>::max();
}

DIAGNOSTIC_PUSH
DIAGNOSTIC_IGNORE
uint32_t getContextLength(struct gguf_file* file) {
  size_t name_idx = ggufFindKey(file, getGGUFKeyMap().at(GGUFKeyType::GENERAL_ARCHITECTURE).c_str());

  const char* n_context_key =
      stringFormatter(getGGUFKeyMap().at(GGUFKeyType::CONTEXT_LENGTH).c_str(), file->kv[name_idx].value.string);
  size_t context_idx = ggufFindKey(file, n_context_key);
  free(const_cast<char*>(n_context_key));

  GGUF_CHECK_ERROR_EQ(static_cast<int>(context_idx), (-1));
  return file->kv[context_idx].value.uint32;

exit:
  return std::numeric_limits<uint32_t>::max();
}

uint32_t getNumDecoders(struct gguf_file* file) {
  size_t name_idx = ggufFindKey(file, getGGUFKeyMap().at(GGUFKeyType::GENERAL_ARCHITECTURE).c_str());

  const char* n_layer_key =
      stringFormatter(getGGUFKeyMap().at(GGUFKeyType::BLOCK_COUNT).c_str(), file->kv[name_idx].value.string);
  size_t layer_idx = ggufFindKey(file, n_layer_key);
  free(const_cast<char*>(n_layer_key));

  GGUF_CHECK_ERROR_EQ(static_cast<int>(layer_idx), (-1));
  return file->kv[layer_idx].value.uint32;

exit:
  return std::numeric_limits<uint32_t>::max();
}

uint32_t getEmbdDim(struct gguf_file* file) {
  size_t name_idx = ggufFindKey(file, getGGUFKeyMap().at(GGUFKeyType::GENERAL_ARCHITECTURE).c_str());

  const char* n_embd_key =
      stringFormatter(getGGUFKeyMap().at(GGUFKeyType::EMBEDDING_LENGTH).c_str(), file->kv[name_idx].value.string);
  size_t embd_idx = ggufFindKey(file, n_embd_key);
  free(const_cast<char*>(n_embd_key));

  GGUF_CHECK_ERROR_EQ(static_cast<int>(embd_idx), (-1));
  return file->kv[embd_idx].value.uint32;

exit:
  return std::numeric_limits<uint32_t>::max();
}

uint32_t getNumHeads(struct gguf_file* file) {
  size_t name_idx = ggufFindKey(file, getGGUFKeyMap().at(GGUFKeyType::GENERAL_ARCHITECTURE).c_str());

  const char* n_head_key =
      stringFormatter(getGGUFKeyMap().at(GGUFKeyType::ATTENTION_HEAD_COUNT).c_str(), file->kv[name_idx].value.string);
  size_t head_idx = ggufFindKey(file, n_head_key);
  free(const_cast<char*>(n_head_key));

  GGUF_CHECK_ERROR_EQ(static_cast<int>(head_idx), (-1));
  return file->kv[head_idx].value.uint32;

exit:
  return std::numeric_limits<uint32_t>::max();
}

uint32_t getNumKVHeads(struct gguf_file* file) {
  size_t name_idx = ggufFindKey(file, getGGUFKeyMap().at(GGUFKeyType::GENERAL_ARCHITECTURE).c_str());

  const char* n_kv_head_key = stringFormatter(getGGUFKeyMap().at(GGUFKeyType::ATTENTION_HEAD_COUNT_KV).c_str(),
                                              file->kv[name_idx].value.string);
  size_t kv_head_idx        = ggufFindKey(file, n_kv_head_key);
  free(const_cast<char*>(n_kv_head_key));

  GGUF_CHECK_ERROR_EQ(static_cast<int>(kv_head_idx), (-1));
  return file->kv[kv_head_idx].value.uint32;

exit:
  return getNumHeads(file);
}

uint32_t getHeadDim(struct gguf_file* file) {
  size_t name_idx = ggufFindKey(file, getGGUFKeyMap().at(GGUFKeyType::GENERAL_ARCHITECTURE).c_str());

  const char* n_head_dim_key =
      stringFormatter(getGGUFKeyMap().at(GGUFKeyType::EMBEDDING_PER_HEAD).c_str(), file->kv[name_idx].value.string);
  size_t head_dim_idx = ggufFindKey(file, n_head_dim_key);
  free(const_cast<char*>(n_head_dim_key));

  GGUF_CHECK_ERROR_EQ(static_cast<int>(head_dim_idx), (-1));
  return file->kv[head_dim_idx].value.uint32;

exit:
  return std::numeric_limits<uint32_t>::max();
}
DIAGNOSTIC_POP

bool getIsCrossAttentionDecoder(struct gguf_file* file) {
  size_t arch_idx = ggufFindKey(file, getGGUFKeyMap().at(GGUFKeyType::GENERAL_ARCHITECTURE).c_str());

  GGUF_CHECK_ERROR_EQ(static_cast<int>(arch_idx), (-1));
  if(file->kv[arch_idx].value.string != nullptr) {
    return !strcmp(GGUF_KEY_DECODER, file->kv[arch_idx].value.string);
  }

exit:
  return false;
}

bool getRopePermute(struct gguf_file* file) {
  size_t rope_layout_idx = ggufFindKey(file, getGGUFKeyMap().at(GGUFKeyType::OPERATION_ROPE_COMPLEX_ORG).c_str());

  GGUF_CHECK_ERROR_EQ(static_cast<int>(rope_layout_idx), (-1));
  return !strcmp("AoS", file->kv[rope_layout_idx].value.string);

exit:
  return false;
}

} // namespace gguf
