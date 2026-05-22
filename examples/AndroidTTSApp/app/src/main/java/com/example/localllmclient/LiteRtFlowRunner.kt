package com.example.llama

import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import java.io.Closeable
import java.io.File

class LiteRtFlowRunner private constructor(
    private val compiledModel: CompiledModel,
    val runtimeMode: String,
    private val modelFile: File,
) : Closeable {
    private val ioLock = Any()
    private val inputBuffers: List<TensorBuffer> = compiledModel.createInputBuffers()
    private val outputBuffers: List<TensorBuffer> = compiledModel.createOutputBuffers()
    private val signatureBindings: SignatureBindings? = createSignatureBindingsOrNull()
    private var bindings: InputBindings? = null
    private val tokenLenHolder = IntArray(1)
    private val promptTokenLenHolder = IntArray(1)
    private val promptFeatLenHolder = IntArray(1)
    private val oneBool = BooleanArray(1)
    private val oneInt = IntArray(1)
    private val oneLong = LongArray(1)
    private var tokenIntScratch = IntArray(0)
    private var promptTokenIntScratch = IntArray(0)

    init {
        require(inputBuffers.size >= 6) { "LiteRT input count < 6: ${inputBuffers.size}" }
        require(outputBuffers.isNotEmpty()) { "LiteRT output count is 0" }
    }

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
        require(token.isNotEmpty()) { "token cannot be empty" }
        require(promptToken.isNotEmpty()) { "promptToken cannot be empty" }
        require(promptFeat.isNotEmpty()) { "promptFeat cannot be empty" }
        require(embedding.isNotEmpty()) { "embedding cannot be empty" }
        synchronized(ioLock) {
            signatureBindings?.let { sb ->
                writeTokenLike(sb.inputs[IN_ARGS_0]!!, token, "args_0")
                tokenLenHolder[0] = tokenLen.coerceAtLeast(1)
                sb.inputs[IN_ARGS_1]!!.writeInt(tokenLenHolder)
                writeTokenLike(sb.inputs[IN_ARGS_2]!!, promptToken, "args_2")
                promptTokenLenHolder[0] = promptTokenLen.coerceAtLeast(1)
                sb.inputs[IN_ARGS_3]!!.writeInt(promptTokenLenHolder)
                sb.inputs[IN_ARGS_4]!!.writeFloat(promptFeat)
                promptFeatLenHolder[0] = promptFeatLen.coerceAtLeast(1)
                sb.inputs[IN_ARGS_5]!!.writeInt(promptFeatLenHolder)
                sb.inputs[IN_ARGS_6]!!.writeFloat(embedding)

                compiledModel.run(sb.inputs, sb.outputs, sb.signatureKey)
                return sb.outputs[OUT_OUTPUT_0]!!.readFloat()
            }

            val resolved = resolveBindingsIfNeeded(
                token = token,
                tokenLen = tokenLen,
                promptToken = promptToken,
                promptTokenLen = promptTokenLen,
                promptFeat = promptFeat,
                promptFeatLen = promptFeatLen,
                embedding = embedding,
                streaming = streaming,
                finalize = finalize,
            )
            writeInputs(
                resolved = resolved,
                token = token,
                tokenLen = tokenLen,
                promptToken = promptToken,
                promptTokenLen = promptTokenLen,
                promptFeat = promptFeat,
                promptFeatLen = promptFeatLen,
                embedding = embedding,
                streaming = streaming,
                finalize = finalize,
            )
            // Keep one explicit synchronous call path: run + output read on the same thread.
            compiledModel.run(inputBuffers, outputBuffers)
            return outputBuffers[0].readFloat()
        }
    }

    override fun close() {
        signatureBindings?.inputs?.values?.forEach { runCatching { it.close() } }
        signatureBindings?.outputs?.values?.forEach { runCatching { it.close() } }
        inputBuffers.forEach { runCatching { it.close() } }
        outputBuffers.forEach { runCatching { it.close() } }
        runCatching { compiledModel.close() }
    }

    private data class SignatureBindings(
        val signatureKey: String,
        val inputs: LinkedHashMap<String, TensorBuffer>,
        val outputs: LinkedHashMap<String, TensorBuffer>,
    )

    private data class InputBindings(
        val token: Int,
        val tokenLen: Int,
        val promptToken: Int,
        val promptTokenLen: Int,
        val promptFeat: Int,
        val embedding: Int,
        val promptFeatLen: Int = -1,
        val streaming: Int = -1,
        val finalize: Int = -1,
    )

    private fun resolveBindingsIfNeeded(
        token: LongArray,
        tokenLen: Int,
        promptToken: LongArray,
        promptTokenLen: Int,
        promptFeat: FloatArray,
        promptFeatLen: Int,
        embedding: FloatArray,
        streaming: Boolean,
        finalize: Boolean,
    ): InputBindings {
        bindings?.let { return it }

        // Prefer deterministic positional mapping for known flow input layouts.
        // 7-input layout:
        //   0 token, 1 token_len, 2 prompt_token, 3 prompt_token_len, 4 prompt_feat, 5 prompt_feat_len, 6 embedding
        // 9-input layout:
        //   above + 7 streaming, 8 finalize
        if (inputBuffers.size == 7 || inputBuffers.size == 9) {
            val resolved =
                InputBindings(
                    token = 0,
                    tokenLen = 1,
                    promptToken = 2,
                    promptTokenLen = 3,
                    promptFeat = 4,
                    promptFeatLen = 5,
                    embedding = 6,
                    streaming = if (inputBuffers.size >= 8) 7 else -1,
                    finalize = if (inputBuffers.size >= 9) 8 else -1,
                )
            bindings = resolved
            Log.i(TAG, "LiteRT positional-bind: inputCount=${inputBuffers.size}, model=${modelFile.name}")
            if (resolved.streaming < 0 || resolved.finalize < 0) {
                Log.w(
                    TAG,
                    "LiteRT optional input missing: streaming=${resolved.streaming}, finalize=${resolved.finalize} for ${modelFile.name}",
                )
            }
            return resolved
        }

        val used = HashSet<Int>()
        val tokenIdx = bindRequired(inputBuffers, used, "token") { writeTokenLike(it, token, "token") }
        val tokenLenIdx = bindRequired(inputBuffers, used, "token_len") {
            tokenLenHolder[0] = tokenLen.coerceAtLeast(1)
            it.writeInt(tokenLenHolder)
        }
        val promptTokenIdx = bindRequired(inputBuffers, used, "prompt_token") {
            writeTokenLike(it, promptToken, "prompt_token")
        }
        val promptTokenLenIdx = bindRequired(inputBuffers, used, "prompt_token_len") {
            promptTokenLenHolder[0] = promptTokenLen.coerceAtLeast(1)
            it.writeInt(promptTokenLenHolder)
        }
        val promptFeatIdx = bindRequired(inputBuffers, used, "prompt_feat") { it.writeFloat(promptFeat) }
        val embeddingIdx = bindRequired(inputBuffers, used, "embedding") { it.writeFloat(embedding) }
        val promptFeatLenIdx = bindOptional(inputBuffers, used, "prompt_feat_len") {
            promptFeatLenHolder[0] = promptFeatLen.coerceAtLeast(1)
            it.writeInt(promptFeatLenHolder)
        }
        val streamingIdx = bindOptional(inputBuffers, used, "streaming") { writeBoolLike(it, streaming, "streaming") }
        val finalizeIdx = bindOptional(inputBuffers, used, "finalize") { writeBoolLike(it, finalize, "finalize") }
        val resolved = InputBindings(
            token = tokenIdx,
            tokenLen = tokenLenIdx,
            promptToken = promptTokenIdx,
            promptTokenLen = promptTokenLenIdx,
            promptFeat = promptFeatIdx,
            embedding = embeddingIdx,
            promptFeatLen = promptFeatLenIdx,
            streaming = streamingIdx,
            finalize = finalizeIdx,
        )
        bindings = resolved
        if (resolved.streaming < 0 || resolved.finalize < 0) {
            Log.w(
                TAG,
                "LiteRT optional input missing: streaming=${resolved.streaming}, finalize=${resolved.finalize} for ${modelFile.name}",
            )
        }
        return resolved
    }

    private fun writeInputs(
        resolved: InputBindings,
        token: LongArray,
        tokenLen: Int,
        promptToken: LongArray,
        promptTokenLen: Int,
        promptFeat: FloatArray,
        promptFeatLen: Int,
        embedding: FloatArray,
        streaming: Boolean,
        finalize: Boolean,
    ) {
        writeTokenLike(inputBuffers[resolved.token], token, "token")
        tokenLenHolder[0] = tokenLen.coerceAtLeast(1)
        inputBuffers[resolved.tokenLen].writeInt(tokenLenHolder)
        writeTokenLike(inputBuffers[resolved.promptToken], promptToken, "prompt_token")
        promptTokenLenHolder[0] = promptTokenLen.coerceAtLeast(1)
        inputBuffers[resolved.promptTokenLen].writeInt(promptTokenLenHolder)
        inputBuffers[resolved.promptFeat].writeFloat(promptFeat)
        inputBuffers[resolved.embedding].writeFloat(embedding)

        if (resolved.promptFeatLen >= 0) {
            promptFeatLenHolder[0] = promptFeatLen.coerceAtLeast(1)
            inputBuffers[resolved.promptFeatLen].writeInt(promptFeatLenHolder)
        }
        if (resolved.streaming >= 0) {
            writeBoolLike(inputBuffers[resolved.streaming], streaming, "streaming")
        }
        if (resolved.finalize >= 0) {
            writeBoolLike(inputBuffers[resolved.finalize], finalize, "finalize")
        }
    }

    private fun bindRequired(
        buffers: List<TensorBuffer>,
        used: MutableSet<Int>,
        label: String,
        writer: (TensorBuffer) -> Unit,
    ): Int {
        for (i in buffers.indices) {
            if (used.contains(i)) continue
            if (runCatching { writer(buffers[i]) }.isSuccess) {
                used.add(i)
                Log.i(TAG, "LiteRT auto-bind: $label -> input[$i]")
                return i
            }
        }
        error("LiteRT auto-bind failed for required input: $label")
    }

    private fun bindOptional(
        buffers: List<TensorBuffer>,
        used: MutableSet<Int>,
        label: String,
        writer: (TensorBuffer) -> Unit,
    ): Int {
        for (i in buffers.indices) {
            if (used.contains(i)) continue
            if (runCatching { writer(buffers[i]) }.isSuccess) {
                used.add(i)
                Log.i(TAG, "LiteRT auto-bind(optional): $label -> input[$i]")
                return i
            }
        }
        return -1
    }

    private fun writeTokenLike(buffer: TensorBuffer, value: LongArray, label: String) {
        runCatching {
            val intArray =
                if (label == "token") {
                    if (tokenIntScratch.size != value.size) tokenIntScratch = IntArray(value.size)
                    tokenIntScratch
                } else {
                    if (promptTokenIntScratch.size != value.size) promptTokenIntScratch = IntArray(value.size)
                    promptTokenIntScratch
                }
            var i = 0
            while (i < value.size) {
                val v = value[i]
                require(v in Int.MIN_VALUE.toLong()..Int.MAX_VALUE.toLong()) {
                    "$label token[$i]=$v overflows int32"
                }
                intArray[i] = v.toInt()
                i += 1
            }
            buffer.writeInt(intArray)
            return
        }.onFailure {
            buffer.writeLong(value)
            Log.w(TAG, "LiteRT input '$label' accepted int64 instead of int32 for ${modelFile.name}")
        }
    }

    private fun writeBoolLike(buffer: TensorBuffer, value: Boolean, label: String) {
        runCatching {
            oneBool[0] = value
            buffer.writeBoolean(oneBool)
            return
        }
        runCatching {
            oneInt[0] = if (value) 1 else 0
            buffer.writeInt(oneInt)
            Log.w(TAG, "LiteRT input '$label' accepted int32 instead of bool for ${modelFile.name}")
            return
        }
        oneLong[0] = if (value) 1L else 0L
        buffer.writeLong(oneLong)
        Log.w(TAG, "LiteRT input '$label' accepted int64 instead of bool for ${modelFile.name}")
    }

    private fun createSignatureBindingsOrNull(): SignatureBindings? {
        if (inputBuffers.size != 7) return null
        return runCatching {
            val inputs = linkedMapOf<String, TensorBuffer>()
            inputs[IN_ARGS_0] = compiledModel.createInputBuffer(IN_ARGS_0, SIGNATURE_KEY)
            inputs[IN_ARGS_1] = compiledModel.createInputBuffer(IN_ARGS_1, SIGNATURE_KEY)
            inputs[IN_ARGS_2] = compiledModel.createInputBuffer(IN_ARGS_2, SIGNATURE_KEY)
            inputs[IN_ARGS_3] = compiledModel.createInputBuffer(IN_ARGS_3, SIGNATURE_KEY)
            inputs[IN_ARGS_4] = compiledModel.createInputBuffer(IN_ARGS_4, SIGNATURE_KEY)
            inputs[IN_ARGS_5] = compiledModel.createInputBuffer(IN_ARGS_5, SIGNATURE_KEY)
            inputs[IN_ARGS_6] = compiledModel.createInputBuffer(IN_ARGS_6, SIGNATURE_KEY)

            val outputs = linkedMapOf<String, TensorBuffer>()
            outputs[OUT_OUTPUT_0] = compiledModel.createOutputBuffer(OUT_OUTPUT_0, SIGNATURE_KEY)

            Log.i(
                TAG,
                "LiteRT signature-bind enabled: signature=$SIGNATURE_KEY, inputs=${inputs.keys}, outputs=${outputs.keys}",
            )
            SignatureBindings(
                signatureKey = SIGNATURE_KEY,
                inputs = inputs,
                outputs = outputs,
            )
        }.onFailure { t ->
            Log.w(TAG, "LiteRT signature-bind unavailable, fallback to positional/auto bind: ${t.message}")
        }.getOrNull()
    }

    companion object {
        private const val TAG = "LiteRtFlowRunner"
        private const val SIGNATURE_KEY = "serving_default"
        private const val IN_ARGS_0 = "args_0"
        private const val IN_ARGS_1 = "args_1"
        private const val IN_ARGS_2 = "args_2"
        private const val IN_ARGS_3 = "args_3"
        private const val IN_ARGS_4 = "args_4"
        private const val IN_ARGS_5 = "args_5"
        private const val IN_ARGS_6 = "args_6"
        private const val OUT_OUTPUT_0 = "output_0"
        private const val MAX_SAFE_MODEL_BYTES = 500L * 1024L * 1024L
        // Empirical guardrail:
        // On Pixel 10 Pro (PowerVR OpenCL stack), compiling the ~466 MiB flow.tflite
        // crashes inside libPVROCL with Scudo OOM during kernel creation.
        // Keep OpenCL only for smaller models.
        private data class GpuAttempt(
            val mode: String,
            val optionsBuilder: () -> CompiledModel.Options,
        )

        fun load(modelFile: File): LiteRtFlowRunner {
            require(modelFile.exists() && modelFile.isFile) {
                "LiteRT flow model not found: ${modelFile.absolutePath}"
            }
            val modelBytes = modelFile.length()
            require(modelBytes in 1..MAX_SAFE_MODEL_BYTES) {
                "LiteRT flow model too large for stable GPU compile on this device: " +
                    "${modelFile.name} (${modelBytes} bytes). " +
                    "Please replace with a smaller/optimized flow.tflite."
            }

            val attempts = mutableListOf<GpuAttempt>()
            attempts += GpuAttempt("GPU(OPENCL,FP32)") {
                CompiledModel.Options(Accelerator.GPU).apply {
                    this.gpuOptions = CompiledModel.GpuOptions(
                        precision = CompiledModel.GpuOptions.Precision.FP32,
                        backend = CompiledModel.GpuOptions.Backend.OPENCL,
                        numStepsOfCommandBufferPreparations = 0,
                    )
                }
            }
            attempts += GpuAttempt("GPU(OPENGL,FP32)") {
                CompiledModel.Options(Accelerator.GPU).apply {
                    this.gpuOptions = CompiledModel.GpuOptions(
                        precision = CompiledModel.GpuOptions.Precision.FP32,
                        backend = CompiledModel.GpuOptions.Backend.OPENGL,
                        numStepsOfCommandBufferPreparations = 0,
                    )
                }
            }
            attempts += GpuAttempt("GPU(AUTOMATIC,FP32)") {
                CompiledModel.Options(Accelerator.GPU).apply {
                    this.gpuOptions = CompiledModel.GpuOptions(
                        precision = CompiledModel.GpuOptions.Precision.FP32,
                        backend = CompiledModel.GpuOptions.Backend.AUTOMATIC,
                        numStepsOfCommandBufferPreparations = 0,
                    )
                }
            }

            attempts.forEach { attempt ->
                val result = runCatching {
                    Log.i(TAG, "Try LiteRT GPU mode=${attempt.mode}")
                    CompiledModel.create(modelFile.absolutePath, attempt.optionsBuilder())
                }
                val compiled = result.getOrNull()
                if (compiled != null) {
                    Log.i(TAG, "LiteRT flow model loaded in GPU mode=${attempt.mode}: ${modelFile.absolutePath}")
                    return LiteRtFlowRunner(
                        compiledModel = compiled,
                        runtimeMode = attempt.mode,
                        modelFile = modelFile,
                    )
                }
                Log.w(TAG, "LiteRT GPU mode failed: ${attempt.mode}, reason=${result.exceptionOrNull()?.message}")
            }

            error("All LiteRT GPU attempts failed for model=${modelFile.name}")
        }
    }
}
