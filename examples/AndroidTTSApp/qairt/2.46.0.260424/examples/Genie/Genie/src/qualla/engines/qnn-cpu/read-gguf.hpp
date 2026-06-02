//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <string>

namespace gguf {

struct gguf_file;

bool ggufFileRead(const char* file_name, struct gguf_file** file);

std::string ggufFilePrint(struct gguf_file* file);

void ggufFileFree(struct gguf_file* f);

uint32_t getContextLength(struct gguf_file* file);

uint32_t getNumDecoders(struct gguf_file* file);

uint32_t getEmbdDim(struct gguf_file* file);

uint32_t getNumHeads(struct gguf_file* file);

uint32_t getNumKVHeads(struct gguf_file* file);

uint32_t getHeadDim(struct gguf_file* file);

bool getRopePermute(struct gguf_file* file);

bool getIsCrossAttentionDecoder(struct gguf_file* file);

} // namespace gguf

