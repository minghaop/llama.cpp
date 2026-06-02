# LiteRT Google Tensor NPU AOT for `flow.tflite`

## Why UI shows `runtime=NPU` but still ~190s
`runtime=NPU` only means NPU path was requested. If compiler plugin / dispatch runtime is missing, LiteRT can still fail back to non-NPU execution, which keeps latency high.

Typical failure logs:
- `Failed to apply compiler plugins: No compiler plugin found`
- `No dispatch library found ...`

## Host requirements for AOT compile
- Linux x86_64
- CPU with AVX
- `GOOGLE_TENSOR_SDK_BETA` points to `litert_plugin_compiler.tar.gz`

## Compile command (official flow wrapper)
```bash
export GOOGLE_TENSOR_SDK_BETA=/abs/path/litert_plugin_compiler.tar.gz
./tools/run_litert_aot_google_tensor.sh \
  --model app/src/main/assets/models/flow.tflite \
  --out-dir npu_aot_out_flow \
  --model-name flow \
  --soc G5
```

## Output artifacts (example)
- `flow_Google_Tensor_G5.tflite`
- `flow_fallback.tflite`
- `flow_aot_report.txt`

## Android integration in this repo
App now prefers compiled Tensor models automatically in this order:
1. `flow_Google_Tensor_G5.tflite`
2. `flow_Google_Tensor_G4.tflite`
3. `flow_Google_Tensor_G3.tflite`
4. `flow_fallback.tflite`
5. `flow.tflite`

So after generating artifacts, place them under:
- `app/src/main/assets/models/` (or runtime synced `files/models/`)
then rebuild/install app.

## Qualcomm AOT (new helper)
For Qualcomm, this repo now also provides a wrapper:

```bash
./tools/run_litert_aot_qualcomm.sh \
  --model app/src/main/assets/models/flow.tflite \
  --out-dir npu_aot_out_flow_qnn \
  --model-name flow \
  --soc SM8750
```

Notes:
- Run on Linux x86_64 (Ubuntu 22.04 recommended by official docs).
- To list SoCs available in your installed LiteRT package:
  `python3 tools/litert_aot_compile_qualcomm.py --list-soc`

## On-device NPU JIT in this app
This project is now configured to test LiteRT Flow inference with NPU JIT first:

- `FLOW_ONLY_TEST_MODE=true`
- `FLOW_ONLY_BACKEND=LITERT`
- `LITERT_FORCE_NPU_JIT_MODEL=true` (prefers `flow.tflite` over precompiled AOT artifacts)

Runtime notes:
- `LiteRtFlowRunner` uses `CompiledModel.Options(Accelerator.NPU)`.
- NPU library readiness is logged; if unavailable, model load fails fast.

Useful logcat filters:
```bash
adb logcat | grep -Ei "LlamaInferenceBridge|LiteRtFlowRunner|FlowOnly\\]\\[LiteRT|NPU|QNN|fallback"
```
