# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from qairt import is_oelinux_user
from qairt.gen_ai_api.configs.gen_ai_config import GenAIConfig
from qairt.gen_ai_api.containers.llm_container import LLMContainer
from qairt.gen_ai_api.executors.t2t_executor import T2TExecutor

if not is_oelinux_user():
    # Builder APIs require converter/compiler dependencies not available on OE-Linux
    from qairt.gen_ai_api.builders.gen_ai_builder import GenAIBuilder
    from qairt.gen_ai_api.builders.gen_ai_builder_htp import GenAIBuilderHTP
    from qairt.gen_ai_api.gen_ai_builder_factory import GenAIBuilderFactory, SupportedLLMs
