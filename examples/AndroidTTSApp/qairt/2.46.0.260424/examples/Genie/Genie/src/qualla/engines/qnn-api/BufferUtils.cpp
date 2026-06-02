//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include "BufferUtils.hpp"
#include "QnnTypeMacros.hpp"
#include "QnnTypeUtils.hpp"
#include "PAL/FileOp.hpp"

namespace aiswutility {

#if !defined(__arm__)
uint32_t calculateByteLength(const std::vector<uint32_t>& dims, const Qnn_DataType_t dataType) {
  uint32_t length = static_cast<uint32_t>(getDataTypeSize(dataType));
  length *= calculateElementCount(dims);
  return length;
}
#endif
size_t calculateByteLength(const std::vector<size_t>& dims, const Qnn_DataType_t dataType) {
  size_t length = static_cast<size_t>(getDataTypeSize(dataType));
  length *= calculateElementCount(dims);
  return length;
}

#if !defined(__arm__)
uint32_t calculateByteLength(const std::vector<uint32_t>& dims,
                             const Qnn_DataType_t dataType,
                             bool& success) {
  uint32_t length = calculateByteLength(dims, dataType);
  if (length == 0) {  // half byte sizes will have their length be 0. Unrecognized datatypes also
                      // have a length of 0.
    success = false;
    return length;
  }
  success = true;
  return length;
}
#endif
size_t calculateByteLength(const std::vector<size_t>& dims,
                           const Qnn_DataType_t dataType,
                           bool& success) {
  size_t length = calculateByteLength(dims, dataType);
  if (length == 0) {  // half byte sizes will have their length be 0. Unrecognized datatypes also
                      // have a length of 0.
    success = false;
    return length;
  }
  success = true;
  return length;
}

#if !defined(__arm__)
float calculateByteExactLength(const std::vector<uint32_t>& dims, const Qnn_DataType_t dataType) {
  return calculateElementCount(dims) * getDataTypeSize(dataType);
}
#endif
float calculateByteExactLength(const std::vector<size_t>& dims, const Qnn_DataType_t dataType) {
  return calculateElementCount(dims) * getDataTypeSize(dataType);
}

#if !defined(__arm__)
uint32_t calculateElementCount(const std::vector<uint32_t>& dims) {
  return static_cast<uint32_t>(
      std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<uint32_t>()));
}
#endif
size_t calculateElementCount(const std::vector<size_t>& dims) {
  return static_cast<size_t>(
      std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>()));
}

uint64_t queryTensorSize(Qnn_Tensor_t tensor) {
  uint32_t* const dims      = QNN_TENSOR_GET_DIMENSIONS(tensor);
  const uint32_t dataFormat = QNN_TENSOR_GET_DATA_FORMAT(tensor);
  return (uint64_t)getBufferSize(dims, dataFormat);
}

uint32_t getBufferSize(const uint32_t* dims, const uint32_t dataFormat) {
  uint32_t height     = dims[1];
  uint32_t width      = dims[2];
  uint32_t bufferSize = 0;
  switch (dataFormat) {
    case QNN_TENSOR_DATA_FORMAT_UBWC_RGBA8888: {
      // Metadata
      bufferSize = DATA_FMT_ALIGN(
          DATA_FMT_ALIGN(((width + 16 - 1) / 16), 64) * DATA_FMT_ALIGN(((height + 4 - 1) / 4), 16),
          4096);
      // Compressed
      bufferSize +=
          DATA_FMT_ALIGN(DATA_FMT_ALIGN(width, 64) * 4 * DATA_FMT_ALIGN(height, 16), 4096);
      break;
    }
    case QNN_TENSOR_DATA_FORMAT_UBWC_NV12_UV: {
      /* adjust h and w to account for full tensor size*/
      width *= 2;
      height *= 2;
    }
    case QNN_TENSOR_DATA_FORMAT_UBWC_NV12:
    case QNN_TENSOR_DATA_FORMAT_UBWC_NV12_Y: {
      // Plane 0 (Y plane)
      // Metadata
      bufferSize = DATA_FMT_ALIGN(
          DATA_FMT_ALIGN(((width + 32 - 1) / 32), 64) * DATA_FMT_ALIGN(((height + 8 - 1) / 8), 16),
          4096);
      // Compressed
      bufferSize += DATA_FMT_ALIGN(DATA_FMT_ALIGN(width, 128) * DATA_FMT_ALIGN(height, 32), 4096);
      // Plane 1 (UV plane)
      // Metadata
      bufferSize += DATA_FMT_ALIGN(DATA_FMT_ALIGN(((width / 2 + 16 - 1) / 16), 64) *
                                       DATA_FMT_ALIGN(((height / 2 + 8 - 1) / 8), 16),
                                   4096);
      // Compressed
      bufferSize +=
          DATA_FMT_ALIGN(DATA_FMT_ALIGN(width / 2, 64) * 2 * DATA_FMT_ALIGN(height / 2, 32), 4096);
      break;
    }
    case QNN_TENSOR_DATA_FORMAT_UBWC_NV124R_UV: {
      /* adjust h and w to account for full tensor size*/
      width *= 2;
      height *= 2;
    }
    case QNN_TENSOR_DATA_FORMAT_UBWC_NV124R:
    case QNN_TENSOR_DATA_FORMAT_UBWC_NV124R_Y: {
      // Plane 0 (Y plane)
      // Metadata
      bufferSize = DATA_FMT_ALIGN(
          DATA_FMT_ALIGN(((width + 64 - 1) / 64), 64) * DATA_FMT_ALIGN(((height + 4 - 1) / 4), 16),
          4096);
      // Compressed
      bufferSize += DATA_FMT_ALIGN(DATA_FMT_ALIGN(width, 256) * DATA_FMT_ALIGN(height, 16), 4096);
      // Plane 1 (UV plane)
      // Metadata
      bufferSize += DATA_FMT_ALIGN(DATA_FMT_ALIGN(((width / 2 + 32 - 1) / 32), 64) *
                                       DATA_FMT_ALIGN(((height / 2 + 4 - 1) / 4), 16),
                                   4096);
      // Compressed
      bufferSize +=
          DATA_FMT_ALIGN(DATA_FMT_ALIGN(width / 2, 128) * 2 * DATA_FMT_ALIGN(height / 2, 16), 4096);
      break;
    }
    default:
      return bufferSize;
  }
  return bufferSize;
}

static uint32_t readLe32(const uint8_t* p)
{
  const uint32_t b0 = static_cast<uint32_t>(p[0]);
  const uint32_t b1 = static_cast<uint32_t>(p[1]);
  const uint32_t b2 = static_cast<uint32_t>(p[2]);
  const uint32_t b3 = static_cast<uint32_t>(p[3]);
  return (b0 |
          (b1 << 8U) |
          (b2 << 16U) |
          (b3 << 24U));
}

static uint16_t readLe16(const uint8_t* p)
{
  const uint32_t b0 = static_cast<uint32_t>(p[0]);
  const uint32_t b1 = static_cast<uint32_t>(p[1]);
  const uint32_t v = (b0 | (b1 << 8U));
  return static_cast<uint16_t>(v & 0xFFFFU);
}

bool isDlcBuffer(const uint8_t* buffer, size_t size)
{
  if ((buffer == nullptr) || (size < static_cast<size_t>(28U)))
  {
    return false;
  }

  // ZIP Local File Header signature: 0x04034b50 (little-endian)
  const uint32_t localFileHeaderSignature = readLe32(buffer);
  if (localFileHeaderSignature != 0x04034b50U)
  {
    return false;
  }

  // ZIP version needed to extract is at offset 4 (2 bytes, little-endian)
  const uint16_t versionNeededToExtract = readLe16(&buffer[4]);
  if (versionNeededToExtract != static_cast<uint16_t>(45U))
  {
    return false;
  }

  // EOCD signature is at the start of the last 22 bytes (TBD - might not work for gcc, msvc)
  const size_t eocdOffset = size - static_cast<size_t>(22U);
  const uint32_t endOfCentralDirSignature = readLe32(&buffer[eocdOffset]);
  if (endOfCentralDirSignature != 0x06054b50U)
  {
    return false;
  }

  return true;
}

bool GetFileDataAsBuffer(const std::string& filePath,
                         std::vector<std::uint8_t>& buffer)
{
  const int32_t fd = pal::FileOp::open(filePath, pal::FileOp::AccessMode::O_RDONLY_);
  if (fd == -1)
  {
    // Log here
    return false;
  }

  const std::int64_t fileSize64 = pal::FileOp::totalBytes(fd);
  if (fileSize64 <= 0)
  {
    // Log here
    (void)pal::FileOp::close(fd);
    return false;
  }

  const std::uint64_t fileSizeU64 = static_cast<std::uint64_t>(fileSize64);

  if ( fileSizeU64 > std::numeric_limits<std::size_t>::max())
  {
    // Log here: file too large to allocate on this platform
    (void)pal::FileOp::close(fd);
    return false;
  }

  const std::size_t totalSize = static_cast<std::size_t>(fileSizeU64);

  buffer.resize(totalSize);

  const std::size_t maxChunkSize =
    static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max());

  std::size_t offset = 0U;

  while (offset < totalSize)
  {
    const std::size_t remaining = totalSize - offset;
    const std::size_t chunkSizeSt = (remaining < maxChunkSize) ? remaining : maxChunkSize;
    const std::uint32_t chunkSizeU32 = static_cast<std::uint32_t>(chunkSizeSt);

    const std::int32_t bytesRead32 =
      pal::FileOp::read(fd, static_cast<void*>(&buffer[offset]), chunkSizeU32);

    if (bytesRead32 < 0)
    {
      // Log here: read failure
      (void)pal::FileOp::close(fd);
      return false;
    }

    if (bytesRead32 == 0)
    {
      // Log here: unexpected EOF / no progress
      (void)pal::FileOp::close(fd);
      return false;
    }

    const std::size_t bytesReadSt = static_cast<std::size_t>(bytesRead32);
    offset += bytesReadSt;

    if (bytesReadSt != chunkSizeSt)
    {
      // Log here: short read (matches original "readSize != fileSize" intent)
      (void)pal::FileOp::close(fd);
      return false;
    }
  }

  if (pal::FileOp::close(fd) == -1)
  {
    // Log here
  }

  return true;
}

}  // namespace aiswutility
