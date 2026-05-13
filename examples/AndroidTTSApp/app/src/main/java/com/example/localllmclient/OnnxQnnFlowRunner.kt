package com.example.llama

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.util.Log
import java.io.Closeable
import java.io.File
import java.nio.FloatBuffer
import java.nio.IntBuffer
import java.nio.LongBuffer

/**
 * Flow runner backed by ONNX Runtime QNN Execution Provider (GPU backend).
 *
 * The ONNX model in assets/models expects:
 * - token: int64[1, token_len]
 * - token_len: int32[1]
 * - prompt_token: int64[1, prompt_token_len]
 * - prompt_token_len: int32[1]
 * - prompt_feat: float32[1, 644, prompt_feat_len]
 * - embedding: float32[1, 192]
 */
class OnnxQnnFlowRunner private constructor(
    private val ortEnv: OrtEnvironment,
    private val sessionOptions: OrtSession.SessionOptions,
    private val session: OrtSession,
    val cpuFallbackDisabled: Boolean,
) : Closeable {

    private val tokenInputName: String = resolveInputName(listOf("token"))
    private val tokenLenInputName: String = resolveInputName(listOf("token_len"))
    private val promptTokenInputName: String = resolveInputName(listOf("prompt_token"))
    private val promptTokenLenInputName: String = resolveInputName(listOf("prompt_token_len"))
    private val promptFeatInputName: String = resolveInputName(listOf("prompt_feat"))
    private val embeddingInputName: String = resolveInputName(listOf("embedding"))
    private val outputName: String = session.outputNames.firstOrNull()
        ?: error("ONNX flow model has no outputs")

    fun forward(
        token: LongArray,
        tokenLen: Int,
        promptToken: LongArray,
        promptTokenLen: Int,
        promptFeat: FloatArray,
        promptFeatLen: Int,
        embedding: FloatArray,
        streaming: Boolean,
        finalize: Boolean,
    ): FloatArray {
        check(token.isNotEmpty()) { "token cannot be empty" }
        check(promptToken.isNotEmpty()) { "promptToken cannot be empty" }
        check(embedding.isNotEmpty()) { "embedding cannot be empty" }
        check(promptFeat.isNotEmpty()) { "promptFeat cannot be empty" }

        val inferredPromptFeatLen = promptFeat.size / FLOW_PROMPT_FEAT_DIM
        check(inferredPromptFeatLen * FLOW_PROMPT_FEAT_DIM == promptFeat.size) {
            "promptFeat size ${promptFeat.size} is not divisible by $FLOW_PROMPT_FEAT_DIM"
        }
        val resolvedPromptFeatLen = when {
            promptFeatLen > 0 -> promptFeatLen
            else -> inferredPromptFeatLen
        }
        check(resolvedPromptFeatLen == inferredPromptFeatLen) {
            "promptFeatLen mismatch: arg=$promptFeatLen, inferred=$inferredPromptFeatLen"
        }

        val resolvedTokenLen = tokenLen.coerceAtLeast(1)
        val resolvedPromptTokenLen = promptTokenLen.coerceAtLeast(1)

        // These flags exist in MNN flow path. Current ONNX flow graph does not consume them.
            Log.d(TAG, "forward flags ignored by ONNX flow: streaming=$streaming, finalize=$finalize")

        val tensors = ArrayList<OnnxTensor>(6)
        try {
            val tokenTensor = OnnxTensor.createTensor(
                ortEnv,
                LongBuffer.wrap(token),
                longArrayOf(1, token.size.toLong()),
            )
            tensors.add(tokenTensor)

            val tokenLenTensor = OnnxTensor.createTensor(
                ortEnv,
                IntBuffer.wrap(intArrayOf(resolvedTokenLen)),
                longArrayOf(1),
            )
            tensors.add(tokenLenTensor)

            val promptTokenTensor = OnnxTensor.createTensor(
                ortEnv,
                LongBuffer.wrap(promptToken),
                longArrayOf(1, promptToken.size.toLong()),
            )
            tensors.add(promptTokenTensor)

            val promptTokenLenTensor = OnnxTensor.createTensor(
                ortEnv,
                IntBuffer.wrap(intArrayOf(resolvedPromptTokenLen)),
                longArrayOf(1),
            )
            tensors.add(promptTokenLenTensor)

            val promptFeatTensor = OnnxTensor.createTensor(
                ortEnv,
                FloatBuffer.wrap(promptFeat),
                longArrayOf(1, FLOW_PROMPT_FEAT_DIM.toLong(), resolvedPromptFeatLen.toLong()),
            )
            tensors.add(promptFeatTensor)

            val embeddingTensor = OnnxTensor.createTensor(
                ortEnv,
                FloatBuffer.wrap(embedding),
                longArrayOf(1, embedding.size.toLong()),
            )
            tensors.add(embeddingTensor)

            val inputs = HashMap<String, OnnxTensor>(6)
            inputs[tokenInputName] = tokenTensor
            inputs[tokenLenInputName] = tokenLenTensor
            inputs[promptTokenInputName] = promptTokenTensor
            inputs[promptTokenLenInputName] = promptTokenLenTensor
            inputs[promptFeatInputName] = promptFeatTensor
            inputs[embeddingInputName] = embeddingTensor

            session.run(inputs, setOf(outputName)).use { outputs ->
                return flattenToFloatArray(outputs[0].value)
            }
        } finally {
            tensors.forEach { tensor ->
                runCatching { tensor.close() }
            }
        }
    }

    override fun close() {
        runCatching { session.close() }
        runCatching { sessionOptions.close() }
    }

    private fun resolveInputName(candidates: List<String>): String {
        val names = session.inputNames.toList()
        candidates.forEach { target ->
            names.firstOrNull { it.equals(target, ignoreCase = true) }?.let { return it }
        }
        candidates.forEach { target ->
            names.firstOrNull { it.contains(target, ignoreCase = true) }?.let { return it }
        }
        error("ONNX flow model missing input, expected one of $candidates, actual=$names")
    }

    companion object {
        private const val TAG = "OnnxQnnFlowRunner"
        private const val FLOW_PROMPT_FEAT_DIM = 644

        fun load(
            modelFile: File,
            backendPath: String = DEFAULT_QNN_GPU_BACKEND_PATH,
            disableCpuFallback: Boolean = true,
        ): OnnxQnnFlowRunner {
            require(modelFile.exists() && modelFile.isFile) {
                "ONNX flow model not found: ${modelFile.absolutePath}"
            }

            val ortEnv = OrtEnvironment.getEnvironment()
            val options = OrtSession.SessionOptions()
            try {
                options.setIntraOpNumThreads(1)
                options.setInterOpNumThreads(1)
                if (disableCpuFallback) {
                    options.addConfigEntry("session.disable_cpu_ep_fallback", "1")
                }

                val qnnOptions = linkedMapOf(
                    "backend_path" to backendPath,
                )
                options.addQnn(qnnOptions)

                val availableProviders = runCatching { OrtEnvironment.getAvailableProviders() }
                    .getOrNull()
                Log.i(TAG, "ORT available providers: $availableProviders")
                val session = ortEnv.createSession(modelFile.absolutePath, options)
                Log.i(
                    TAG,
                    "ONNX QNN flow session ready: model=${modelFile.absolutePath}, " +
                        "backend=$backendPath, disableCpuFallback=$disableCpuFallback",
                )
                return OnnxQnnFlowRunner(
                    ortEnv = ortEnv,
                    sessionOptions = options,
                    session = session,
                    cpuFallbackDisabled = disableCpuFallback,
                )
            } catch (t: Throwable) {
                runCatching { options.close() }
                throw t
            }
        }


        private fun flattenToFloatArray(value: Any?): FloatArray {
            val out = ArrayList<Float>(1024)
            appendToFloatList(value, out)
            return FloatArray(out.size) { i -> out[i] }
        }

        private fun appendToFloatList(value: Any?, out: MutableList<Float>) {
            when (value) {
                null -> Unit
                is FloatArray -> value.forEach { out.add(it) }
                is DoubleArray -> value.forEach { out.add(it.toFloat()) }
                is Array<*> -> value.forEach { appendToFloatList(it, out) }
                is Number -> out.add(value.toFloat())
                else -> error("Unsupported ONNX output value type: ${value::class.java.name}")
            }
        }
    }
}

internal const val DEFAULT_QNN_GPU_BACKEND_PATH = "libQnnGpu.so"
