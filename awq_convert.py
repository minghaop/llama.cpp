import os
import subprocess
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = '/mist/yangkun/code/llama.cpp/models/Qwen3-4B-Instruct-2507/'
quant_path = '/mist/yangkun/code/llama.cpp/models/Qwen3-4B-Instruct-2507-AWQ/'
llama_cpp_path = '/mist/yangkun/code/llama.cpp'
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }


model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

model.quantize(
    tokenizer,
    quant_config=quant_config,
    export_compatible=True
)

model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
print(f'Model is quantized and saved at "{quant_path}"')



