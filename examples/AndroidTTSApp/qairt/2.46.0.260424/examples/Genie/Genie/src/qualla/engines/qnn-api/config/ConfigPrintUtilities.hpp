//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <ostream>

#ifdef QUALLA_ENGINE_QNN_HTP
#include "QnnHtpCommon.h"
#include "QnnHtpContext.h"
#else
#include "QnnContext.h"
#endif  // QUALLA_ENGINE_QNN_HTP
#include "QnnTypes.h"

std::ostream& operator<<(std::ostream& os, const QnnContext_Config_t& config);
