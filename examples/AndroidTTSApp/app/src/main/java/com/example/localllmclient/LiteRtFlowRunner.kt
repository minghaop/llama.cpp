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
    private val inputBuffers: List<TensorBuffer>
    private val outputBuffers: List<TensorBuffer>
    private val runSignatureKey: String?
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
        val signatureInputs = runCatching { compiledModel.createInputBuffers(SIGNATURE_KEY) }.getOrNull()
        val signatureOutputs = runCatching { compiledModel.createOutputBuffers(SIGNATURE_KEY) }.getOrNull()
        if (!signatureInputs.isNullOrEmpty() && !signatureOutputs.isNullOrEmpty()) {
            inputBuffers = signatureInputs
            outputBuffers = signatureOutputs
            runSignatureKey = SIGNATURE_KEY
            Log.i(TAG, "LiteRT using signature buffers: key=$SIGNATURE_KEY, in=${inputBuffers.size}, out=${outputBuffers.size}")
        } else {
            inputBuffers = compiledModel.createInputBuffers()
            outputBuffers = compiledModel.createOutputBuffers()
            runSignatureKey = null
            Log.i(TAG, "LiteRT using positional buffers: in=${inputBuffers.size}, out=${outputBuffers.size}")
        }
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
            if (runSignatureKey != null) {
                compiledModel.run(inputBuffers, outputBuffers, runSignatureKey)
            } else {
                compiledModel.run(inputBuffers, outputBuffers)
            }
            return outputBuffers[0].readFloat()
        }
    }

    override fun close() {
        inputBuffers.forEach { runCatching { it.close() } }
        outputBuffers.forEach { runCatching { it.close() } }
        runCatching { compiledModel.close() }
    }

    private data class InputBindings(
        val token: Int,
        val tokenLen: Int = -1,
        val promptToken: Int,
        val promptTokenLen: Int = -1,
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

        val used = HashSet<Int>()
        val tokenIdx = bindRequired(inputBuffers, used, "token") { writeTokenLike(it, token, "token") }
        val tokenLenIdx = bindOptional(inputBuffers, used, "token_len") {
            tokenLenHolder[0] = tokenLen.coerceAtLeast(1)
            it.writeInt(tokenLenHolder)
        }
        val promptTokenIdx = bindRequired(inputBuffers, used, "prompt_token") {
            writeTokenLike(it, promptToken, "prompt_token")
        }
        val promptTokenLenIdx = bindOptional(inputBuffers, used, "prompt_token_len") {
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
        if (resolved.tokenLen < 0 || resolved.promptTokenLen < 0 || resolved.promptFeatLen < 0) {
            Log.w(
                TAG,
                "LiteRT length input missing: token_len=${resolved.tokenLen}, " +
                    "prompt_token_len=${resolved.promptTokenLen}, prompt_feat_len=${resolved.promptFeatLen} for ${modelFile.name}",
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
        if (resolved.tokenLen >= 0) {
            tokenLenHolder[0] = tokenLen.coerceAtLeast(1)
            inputBuffers[resolved.tokenLen].writeInt(tokenLenHolder)
        }
        writeTokenLike(inputBuffers[resolved.promptToken], promptToken, "prompt_token")
        if (resolved.promptTokenLen >= 0) {
            promptTokenLenHolder[0] = promptTokenLen.coerceAtLeast(1)
            inputBuffers[resolved.promptTokenLen].writeInt(promptTokenLenHolder)
        }
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
            buffer.writeLong(value)
            return
        }.onFailure {
            Log.w(TAG, "LiteRT input '$label' fallback to int32 write for ${modelFile.name}: ${it.message}")
        }
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
            throw IllegalStateException("LiteRT cannot write token input '$label' for ${modelFile.name}", it)
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

    companion object {
        private const val TAG = "LiteRtFlowRunner"
        private const val SIGNATURE_KEY = "serving_default"
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
            attempts += GpuAttempt("GPU(OPENGL,FP32)") {
                CompiledModel.Options(Accelerator.GPU).apply {
                    this.gpuOptions = CompiledModel.GpuOptions(
                        constantTensorSharing = false,
                        allowSrcQuantizedFcConvOps = false,
                        precision = CompiledModel.GpuOptions.Precision.FP32,
                        bufferStorageType = CompiledModel.GpuOptions.BufferStorageType.BUFFER,
                        preferTextureWeights = false,
                        serializeProgramCache = false,
                        serializeExternalTensors = false,
                        externalTensorsMode = false,
                        backend = CompiledModel.GpuOptions.Backend.OPENGL,
                        priority = CompiledModel.GpuOptions.Priority.HIGH,
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
