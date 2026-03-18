package com.example.llama

import android.content.Context
import com.arm.aichat.AiChat
import com.arm.aichat.InferenceEngine
import com.arm.aichat.isModelLoaded
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Locale
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin

enum class InferenceStage {
    frontEnd,
    llmPrepare,
    llm,
    flow,
    hift,
    voiceGeneration
}

data class StageEndedInfo(
    val stage: InferenceStage,
    val unitName: String,
    val units: Int,
    val seconds: Double,
    val avgUPS: Double,
)

sealed class InferenceEvent {
    data class StageBegan(val stage: InferenceStage, val unitName: String) : InferenceEvent()
    data class StageProgress(
        val stage: InferenceStage,
        val unitName: String,
        val unitsDone: Int,
        val secondsElapsed: Double,
        val instUPS: Double,
        val avgUPS: Double,
    ) : InferenceEvent()

    data class StageEnded(val info: StageEndedInfo) : InferenceEvent()
    data class Failed(val stage: InferenceStage, val message: String) : InferenceEvent()
    data class AudioDuration(val seconds: Double, val samples: Int, val sampleRate: Int) : InferenceEvent()
}

class LlamaInferenceBridge(
    context: Context,
) {
    private val appContext = context.applicationContext
    private val engineMutex = Mutex()
    @Volatile
    private var engine: InferenceEngine? = null
    @Volatile
    private var localResources: LocalResourceFiles? = null
    @Volatile
    private var qwenTokenizerService: QwenTokenizerService? = null
    @Volatile
    private var promptFrontEndEngine: PromptFrontEndEngine? = null

    suspend fun runInference(
        ttsText: String,
        promptText: String,
        promptAudio: FloatArray,
        promptSampleRate: Int,
        onEvent: ((InferenceEvent) -> Unit)? = null,
    ): String {
        fun emit(event: InferenceEvent) {
            onEvent?.invoke(event)
        }

        var currentStage = InferenceStage.frontEnd
        return try {
            val resources = ensureLocalResourcesReady()

            currentStage = InferenceStage.frontEnd
            val frontEndResult = runFrontEndInference(
                promptAudio = promptAudio,
                promptSampleRate = promptSampleRate,
                resources = resources,
                onEvent = onEvent,
            )

            currentStage = InferenceStage.llmPrepare
            val loadedEngine = ensureEngineReady(
                modelFile = resources.llmModel,
                flowModelFile = resources.flowModel,
                hiftModelFile = resources.hiftModel,
            )
            val llmResult = runLLMInference(
                ttsText = ttsText,
                promptText = promptText,
                frontEndResult = frontEndResult,
                engine = loadedEngine,
                resources = resources,
                onEvent = onEvent,
            )

            currentStage = InferenceStage.flow
            val flowResult = runFlowInference(
                llmTokens = llmResult.speechTokens,
                frontEndResult = frontEndResult,
                engine = loadedEngine,
                resources = resources,
                onEvent = onEvent,
            )

            currentStage = InferenceStage.hift
            val hiftResult = runHIFTInference(
                flowResult = flowResult,
                engine = loadedEngine,
                onEvent = onEvent,
            )

            currentStage = InferenceStage.voiceGeneration
            runVoiceGeneration(
                hiftResult = hiftResult,
                onEvent = onEvent,
            )
        } catch (t: Throwable) {
            emit(InferenceEvent.Failed(currentStage, t.message ?: t.toString()))
            "false"
        }
    }

    fun close() {
        val activeEngine = engine
        if (activeEngine != null) {
            runCatching { activeEngine.destroy() }
            engine = null
        }
        val activeTokenizer = qwenTokenizerService
        if (activeTokenizer != null) {
            runCatching { activeTokenizer.close() }
            qwenTokenizerService = null
        }
        val activeFrontEnd = promptFrontEndEngine
        if (activeFrontEnd != null) {
            runCatching { activeFrontEnd.close() }
            promptFrontEndEngine = null
        }
        localResources = null
    }

    private suspend fun runFrontEndInference(
        promptAudio: FloatArray,
        promptSampleRate: Int,
        resources: LocalResourceFiles,
        onEvent: ((InferenceEvent) -> Unit)? = null,
    ): FrontEndResult {
        fun emit(event: InferenceEvent) {
            onEvent?.invoke(event)
        }

        val t0 = nowSeconds()
        emit(InferenceEvent.StageBegan(InferenceStage.frontEnd, "FrontEnd preprocessing"))
        val frontEndEngine = ensurePromptFrontEndEngine(resources)
        val artifacts = frontEndEngine.build(promptAudio, promptSampleRate)
        val units = (
            artifacts.speechTokenLen +
                artifacts.speechFeatLen +
                artifacts.speechEmbedding.size
            ).coerceAtLeast(1)

        emit(
            InferenceEvent.StageEnded(
                StageEndedInfo(
                    stage = InferenceStage.frontEnd,
                    unitName = "frontEnd process completed",
                    units = units,
                    seconds = nowSeconds() - t0,
                    avgUPS = 0.0,
                ),
            ),
        )
        return FrontEndResult(
            speechTokenLen = artifacts.speechTokenLen,
            speechFeatLen = artifacts.speechFeatLen,
            speechEmbeddingLen = artifacts.speechEmbedding.size,
            speechTokens = artifacts.speechTokens,
            speechFeat = artifacts.speechFeat,
            speechEmbedding = artifacts.speechEmbedding,
        )
    }

    private suspend fun runLLMInference(
        ttsText: String,
        promptText: String,
        frontEndResult: FrontEndResult,
        engine: InferenceEngine,
        resources: LocalResourceFiles,
        onEvent: ((InferenceEvent) -> Unit)? = null,
    ): LLMResult {
        fun emit(event: InferenceEvent) {
            onEvent?.invoke(event)
        }

        val prepBegin = nowSeconds()
        emit(InferenceEvent.StageBegan(InferenceStage.llmPrepare, "LLM preprocessing"))

        val tokenizerEntries = resources.tokenizerDir.list()?.size ?: 0
        val tokenizer = ensureQwenTokenizerReady(resources.tokenizerDir)
        val ttsTokenIds = tokenizer.encodeSync(ttsText)
        val promptTokenIds = tokenizer.encodeSync(promptText)
        val ttsTokenLength = ttsTokenIds.size
        val promptTokenLength = promptTokenIds.size
        val prepUnits = (
            ttsTokenLength +
                promptTokenLength +
                tokenizerEntries +
                ((resources.llmEmbedTokens.length() + resources.llmDecoderWeight.length() + resources.llmDecoderBias.length()) % 10_000L).toInt()
            ).coerceAtLeast(1)
        emit(
            InferenceEvent.StageEnded(
                StageEndedInfo(
                    stage = InferenceStage.llmPrepare,
                    unitName = "LLM preprocessing completed",
                    units = prepUnits,
                    seconds = nowSeconds() - prepBegin,
                    avgUPS = 0.0,
                ),
            ),
        )

        val llmBegin = nowSeconds()
        emit(
            InferenceEvent.StageBegan(
                InferenceStage.llm,
                "token generation",
            ),
        )
        val ttsIds = ttsTokenIds
        val promptIds = promptTokenIds
        val textIds = IntArray(promptIds.size + ttsIds.size)
        if (promptIds.isNotEmpty()) {
            System.arraycopy(promptIds, 0, textIds, 0, promptIds.size)
        }
        if (ttsIds.isNotEmpty()) {
            System.arraycopy(ttsIds, 0, textIds, promptIds.size, ttsIds.size)
        }

        val textEmb = embedTokens(resources.llmEmbedTokens.absolutePath, textIds, expectedDim = LLM_HIDDEN_SIZE)
        val sosEosEmb = loadCountPrefixedFloatArray(resources.sosEosEmb.absolutePath)
        val taskIdEmb = loadCountPrefixedFloatArray(resources.taskIdEmb.absolutePath)

        val params = ModelParameters()
        params.loadFromBinary(resources.speechEmbeddingWeight.absolutePath, "speech_embedding.weight")
        params.loadFromBinary(resources.llmDecoderWeight.absolutePath, "llm_decoder.weight")
        params.loadFromBinary(resources.llmDecoderBias.absolutePath, "llm_decoder.bias")

        val promptSpeechTokenEmb = if (frontEndResult.speechTokens.isNotEmpty()) {
            params.embedding("speech_embedding.weight", frontEndResult.speechTokens) ?: FloatArray(0)
        } else {
            FloatArray(0)
        }

        var currentInput = concatFloatArrays(sosEosEmb, textEmb, taskIdEmb, promptSpeechTokenEmb)
        val minLen = (ttsIds.size * 2).coerceAtLeast(0)
        val maxLen = (ttsIds.size * 20).coerceAtLeast(1)

        val outTokens = ArrayList<Int>(maxLen.coerceAtMost(2048))
        var nPast = 0
        val rng = SeededRNG(seed = 0L)

        engine.resetKvCache()

        val reportEverySeconds = 1.0
        var lastReportT = llmBegin
        var lastReportCount = 0

        for (i in 0 until maxLen) {
            val llmRes = engine.decodeEmbeddings(currentInput, nPast)
            val totalElements = llmRes.size
            require(totalElements != 0 && totalElements % LLM_HIDDEN_SIZE == 0) {
                "llm_res size $totalElements not multiple of $LLM_HIDDEN_SIZE"
            }

            nPast += (currentInput.size / LLM_HIDDEN_SIZE).coerceAtLeast(1)
            val start = totalElements - LLM_HIDDEN_SIZE
            val lastTokenVector = llmRes.copyOfRange(start, totalElements)
            val logp = params.computeLogProbabilities(lastTokenVector)
            require(logp != null && logp.isNotEmpty()) { "computeLogProbabilities failed" }

            val topId = SamplingAlgorithm.samplingIds(
                logProbs = logp,
                decoderTokens = outTokens.toIntArray(),
                samplingNum = 25,
                ignoreEOS = i < minLen,
                rng = rng,
            )
            if (topId == SamplingAlgorithm.SPEECH_TOKEN_SIZE) {
                break
            }
            if (topId > SamplingAlgorithm.SPEECH_TOKEN_SIZE) {
                continue
            }
            outTokens.add(topId)

            val embeddingVector = params.getEmbeddingRow("speech_embedding.weight", topId)
            require(!(embeddingVector == null || embeddingVector.isEmpty())) { "getEmbeddingRow failed: $topId" }
            currentInput = embeddingVector

            val now = nowSeconds()
            if (now - lastReportT >= reportEverySeconds) {
                val total = outTokens.size
                val elapsed = (now - llmBegin).coerceAtLeast(1e-9)
                val dt = (now - lastReportT).coerceAtLeast(1e-9)
                val dTok = total - lastReportCount
                val inst = dTok / dt
                val avg = total / elapsed
                emit(
                    InferenceEvent.StageProgress(
                        stage = InferenceStage.llm,
                        unitName = "token",
                        unitsDone = total,
                        secondsElapsed = elapsed,
                        instUPS = inst,
                        avgUPS = avg,
                    ),
                )
                lastReportT = now
                lastReportCount = total
            }
        }

        params.clear()

        val llmSeconds = nowSeconds() - llmBegin
        emit(
            InferenceEvent.StageEnded(
                StageEndedInfo(
                    stage = InferenceStage.llm,
                    unitName = "token generation completed",
                    units = outTokens.size,
                    seconds = llmSeconds,
                    avgUPS = if (llmSeconds > 0) outTokens.size / llmSeconds else 0.0,
                ),
            ),
        )

        return LLMResult(
            tokenCount = outTokens.size,
            speechTokens = outTokens.toIntArray(),
        )
    }

    private suspend fun runFlowInference(
        llmTokens: IntArray,
        frontEndResult: FrontEndResult,
        engine: InferenceEngine,
        resources: LocalResourceFiles,
        onEvent: ((InferenceEvent) -> Unit)? = null,
    ): FlowResult {
        fun emit(event: InferenceEvent) {
            onEvent?.invoke(event)
        }

        val t0 = nowSeconds()
        emit(InferenceEvent.StageBegan(InferenceStage.flow, "Flow inference"))

        require(llmTokens.isNotEmpty()) { "Flow input llmTokens is empty" }
        require(frontEndResult.speechFeat.isNotEmpty()) { "Flow input speechFeat is empty" }
        require(frontEndResult.speechEmbedding.isNotEmpty()) { "Flow input speechEmbedding is empty" }
        require(frontEndResult.speechTokenLen > 0) { "Flow prompt token len is invalid: ${frontEndResult.speechTokenLen}" }
        require(frontEndResult.speechFeatLen > 0) { "Flow prompt feat len is invalid: ${frontEndResult.speechFeatLen}" }

        val noise = loadNoisePe(resources.noisePe)
        val mergedToken = IntArray(frontEndResult.speechTokens.size + llmTokens.size)
        if (frontEndResult.speechTokens.isNotEmpty()) {
            System.arraycopy(frontEndResult.speechTokens, 0, mergedToken, 0, frontEndResult.speechTokens.size)
        }
        if (llmTokens.isNotEmpty()) {
            System.arraycopy(llmTokens, 0, mergedToken, frontEndResult.speechTokens.size, llmTokens.size)
        }

        val flowOutput = engine.encodeFlow(
            inputEmbeddings = frontEndResult.speechEmbedding,
            flowFeat = frontEndResult.speechFeat,
            flowToken = mergedToken,
            tokenLen = llmTokens.size,
            promptTokenLen = frontEndResult.speechTokenLen,
            promptFeatLen = frontEndResult.speechFeatLen,
            randNoise = noise.randNoise,
            extendPe = noise.extendPe,
        )

        val units = mergedToken.size.coerceAtLeast(1)
        val seconds = nowSeconds() - t0

        emit(
            InferenceEvent.StageEnded(
                StageEndedInfo(
                    stage = InferenceStage.flow,
                    unitName = "Flow Inference completed",
                    units = units,
                    seconds = seconds,
                    avgUPS = if (seconds > 0) units / seconds else 0.0,
                ),
            ),
        )
        return FlowResult(units = units, output = flowOutput)
    }

    private suspend fun runHIFTInference(
        flowResult: FlowResult,
        engine: InferenceEngine,
        onEvent: ((InferenceEvent) -> Unit)? = null,
    ): HiftResult {
        fun emit(event: InferenceEvent) {
            onEvent?.invoke(event)
        }

        val t0 = nowSeconds()
        emit(InferenceEvent.StageBegan(InferenceStage.hift, "HifiGan inference"))

        require(flowResult.output.isNotEmpty()) { "HIFT input is empty" }

        val hiftOutput = engine.encodeHift(flowResult.output)
        require(hiftOutput.isNotEmpty()) { "HIFT output is empty" }
        val outputSamples = hiftOutput.size
        emit(
            InferenceEvent.StageEnded(
                StageEndedInfo(
                    stage = InferenceStage.hift,
                    unitName = "HifiGan Inference completed",
                    units = outputSamples,
                    seconds = nowSeconds() - t0,
                    avgUPS = 0.0,
                ),
            ),
        )
        return HiftResult(outputSamples = outputSamples, output = hiftOutput)
    }

    private suspend fun runVoiceGeneration(
        hiftResult: HiftResult,
        onEvent: ((InferenceEvent) -> Unit)? = null,
    ): String {
        fun emit(event: InferenceEvent) {
            onEvent?.invoke(event)
        }

        val voiceBegin = nowSeconds()
        emit(InferenceEvent.StageBegan(InferenceStage.voiceGeneration, "spectrogram to waveform"))

        val outDir = File(appContext.filesDir, "generated_audio").apply { mkdirs() }
        val wavFile = File(outDir, "android_${System.currentTimeMillis()}.wav")
        val sampleRate = 24_000
        require(hiftResult.output.isNotEmpty()) { "HIFT output is empty" }

        val audio = reconstructWaveformFromHift(hiftResult.output)
        require(audio.isNotEmpty()) { "HIFT output produced empty waveform" }
        val clamped = FloatArray(audio.size) { i ->
            audio[i].coerceIn(-0.99f, 0.99f)
        }
        writeMonoFloat32Wav(
            audioData = clamped,
            sampleRate = sampleRate,
            file = wavFile,
        )
        emit(
            InferenceEvent.AudioDuration(
                seconds = clamped.size.toDouble() / sampleRate.toDouble(),
                samples = clamped.size,
                sampleRate = sampleRate,
            ),
        )
        emit(
            InferenceEvent.StageEnded(
                StageEndedInfo(
                    stage = InferenceStage.voiceGeneration,
                    unitName = "spectrogram to waveform completed",
                    units = clamped.size,
                    seconds = nowSeconds() - voiceBegin,
                    avgUPS = 0.0,
                ),
            ),
        )
        return wavFile.absolutePath
    }

    private fun loadNoisePe(file: File): NoisePeData {
        val bytes = file.readBytes()
        require(bytes.size >= 8) { "noise_pe.bin too small: ${file.absolutePath}" }
        val bb = java.nio.ByteBuffer.wrap(bytes).order(java.nio.ByteOrder.LITTLE_ENDIAN)

        val randCount = bb.int
        require(randCount > 0) { "Invalid randNoise count in noise_pe.bin: $randCount" }
        val randRequiredBytes = randCount * 4L
        require(bb.remaining().toLong() >= randRequiredBytes + 4L) {
            "noise_pe.bin truncated for rand_noise"
        }
        val randNoise = FloatArray(randCount)
        for (i in 0 until randCount) {
            randNoise[i] = bb.float
        }

        val peCount = bb.int
        require(peCount > 0) { "Invalid extend_pe count in noise_pe.bin: $peCount" }
        val peRequiredBytes = peCount * 4L
        require(bb.remaining().toLong() >= peRequiredBytes) {
            "noise_pe.bin truncated for extend_pe"
        }
        val extendPe = FloatArray(peCount)
        for (i in 0 until peCount) {
            extendPe[i] = bb.float
        }

        return NoisePeData(randNoise = randNoise, extendPe = extendPe)
    }

    private fun reconstructWaveformFromHift(hiftOutput: FloatArray): FloatArray {
        val firstPartCount = hiftOutput.size
        require(firstPartCount >= HIFT_FRAME_FEATURES) { "HIFT output too short: $firstPartCount" }
        val magnitudeRaw = hiftOutput.copyOfRange(0, firstPartCount / 2)
        val phaseRaw = hiftOutput.copyOfRange(firstPartCount / 2, firstPartCount)
        val nFrames = firstPartCount / HIFT_FRAME_FEATURES
        require(nFrames > 0) { "Invalid HIFT nFrames: $nFrames" }
        val expectedPerHalf = nFrames * HIFT_N_FREQ
        require(magnitudeRaw.size >= expectedPerHalf && phaseRaw.size >= expectedPerHalf) {
            "Invalid HIFT output shape: total=$firstPartCount nFrames=$nFrames"
        }

        val real = FloatArray(nFrames * HIFT_N_FREQ)
        val imag = FloatArray(nFrames * HIFT_N_FREQ)

        var i = 0
        while (i < real.size) {
            val magnitude = min(exp(magnitudeRaw[i].toDouble()).toFloat(), 1e2f)
            val phaseSin = sin(phaseRaw[i].toDouble()).toFloat()
            real[i] = magnitude * cos(phaseSin.toDouble()).toFloat()
            imag[i] = magnitude * sin(phaseSin.toDouble()).toFloat()
            i += 1
        }

        return inverseStft(
            realSpec = real,
            imagSpec = imag,
            nFreq = HIFT_N_FREQ,
            nFrames = nFrames,
            nFft = HIFT_N_FFT,
            hopLength = HIFT_HOP_LENGTH,
            center = true,
        )
    }

    private fun inverseStft(
        realSpec: FloatArray,
        imagSpec: FloatArray,
        nFreq: Int,
        nFrames: Int,
        nFft: Int,
        hopLength: Int,
        center: Boolean,
    ): FloatArray {
        if (nFreq <= 0 || nFrames <= 0 || nFft <= 0 || hopLength <= 0) return FloatArray(0)
        if (realSpec.size < nFreq * nFrames || imagSpec.size < nFreq * nFrames) return FloatArray(0)

        val outputLength = (nFrames - 1) * hopLength + nFft
        val output = FloatArray(outputLength)
        val windowSum = FloatArray(outputLength)
        val window = hannWindow(nFft)

        val fullReal = FloatArray(nFft)
        val fullImag = FloatArray(nFft)

        var frame = 0
        while (frame < nFrames) {
            java.util.Arrays.fill(fullReal, 0f)
            java.util.Arrays.fill(fullImag, 0f)

            var k = 0
            while (k < nFreq) {
                val offset = k * nFrames + frame
                fullReal[k] = realSpec[offset]
                fullImag[k] = imagSpec[offset]
                k += 1
            }

            k = 1
            while (k < nFft / 2) {
                fullReal[nFft - k] = fullReal[k]
                fullImag[nFft - k] = -fullImag[k]
                k += 1
            }

            ifftRadix2InPlace(fullReal, fullImag)

            val start = frame * hopLength
            var i = 0
            while (i < nFft) {
                val idx = start + i
                if (idx < outputLength) {
                    val sample = fullReal[i] * window[i]
                    output[idx] += sample
                    windowSum[idx] += window[i] * window[i]
                }
                i += 1
            }
            frame += 1
        }

        var i = 0
        while (i < outputLength) {
            if (windowSum[i] > 1e-8f) {
                output[i] /= windowSum[i]
            }
            i += 1
        }

        if (!center) return output
        val pad = nFft / 2
        val validLength = outputLength - 2 * pad
        if (validLength <= 0) return output
        return output.copyOfRange(pad, pad + validLength)
    }

    private fun hannWindow(length: Int): FloatArray {
        if (length <= 0) return FloatArray(0)
        val out = FloatArray(length)
        val denom = length.toDouble()
        var i = 0
        while (i < length) {
            out[i] = (0.5 - 0.5 * cos(2.0 * Math.PI * i / denom)).toFloat()
            i += 1
        }
        return out
    }

    private fun ifftRadix2InPlace(real: FloatArray, imag: FloatArray) {
        val n = real.size
        require(n == imag.size) { "IFFT input size mismatch" }
        require(n > 0 && (n and (n - 1)) == 0) { "IFFT size must be power of two: $n" }

        var i = 0
        while (i < n) {
            imag[i] = -imag[i]
            i += 1
        }
        fftRadix2InPlace(real, imag)
        val scale = 1f / n.toFloat()
        i = 0
        while (i < n) {
            real[i] *= scale
            imag[i] = -imag[i] * scale
            i += 1
        }
    }

    private fun fftRadix2InPlace(real: FloatArray, imag: FloatArray) {
        val n = real.size
        require(n == imag.size) { "FFT input size mismatch" }
        require(n > 0 && (n and (n - 1)) == 0) { "FFT size must be power of two: $n" }

        var j = 0
        var i = 1
        while (i < n) {
            var bit = n ushr 1
            while ((j and bit) != 0) {
                j = j xor bit
                bit = bit ushr 1
            }
            j = j xor bit
            if (i < j) {
                val tr = real[i]
                real[i] = real[j]
                real[j] = tr
                val ti = imag[i]
                imag[i] = imag[j]
                imag[j] = ti
            }
            i += 1
        }

        var len = 2
        while (len <= n) {
            val halfLen = len ushr 1
            val theta = -2.0 * Math.PI / len.toDouble()
            val wLenR = cos(theta).toFloat()
            val wLenI = sin(theta).toFloat()

            var start = 0
            while (start < n) {
                var wR = 1f
                var wI = 0f
                var k = 0
                while (k < halfLen) {
                    val u = start + k
                    val v = u + halfLen
                    val tR = wR * real[v] - wI * imag[v]
                    val tI = wR * imag[v] + wI * real[v]

                    real[v] = real[u] - tR
                    imag[v] = imag[u] - tI
                    real[u] += tR
                    imag[u] += tI

                    val nextWR = wR * wLenR - wI * wLenI
                    wI = wR * wLenI + wI * wLenR
                    wR = nextWR
                    k += 1
                }
                start += len
            }
            len = len shl 1
        }
    }

    private fun writeMonoFloat32Wav(audioData: FloatArray, sampleRate: Int, file: File) {
        if (file.exists()) {
            file.delete()
        }
        val numChannels = 1
        val bitsPerSample = 32
        val audioFormat = 3 // IEEE float
        val byteRate = sampleRate * numChannels * (bitsPerSample / 8)
        val blockAlign = numChannels * (bitsPerSample / 8)
        val dataSize = audioData.size * 4
        val chunkSize = 36 + dataSize

        FileOutputStream(file).use { out ->
            val header = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
            header.put("RIFF".toByteArray(Charsets.US_ASCII))
            header.putInt(chunkSize)
            header.put("WAVE".toByteArray(Charsets.US_ASCII))
            header.put("fmt ".toByteArray(Charsets.US_ASCII))
            header.putInt(16)
            header.putShort(audioFormat.toShort())
            header.putShort(numChannels.toShort())
            header.putInt(sampleRate)
            header.putInt(byteRate)
            header.putShort(blockAlign.toShort())
            header.putShort(bitsPerSample.toShort())
            header.put("data".toByteArray(Charsets.US_ASCII))
            header.putInt(dataSize)
            out.write(header.array())

            val payload = ByteBuffer.allocate(dataSize).order(ByteOrder.LITTLE_ENDIAN)
            var i = 0
            while (i < audioData.size) {
                payload.putFloat(audioData[i])
                i += 1
            }
            out.write(payload.array())
        }
    }

    private suspend fun ensureEngineReady(modelFile: File, flowModelFile: File, hiftModelFile: File): InferenceEngine =
        engineMutex.withLock {
            engine?.let { return it }

            installBundledResourcesIfPresent()
            val loaded = AiChat.getInferenceEngine(appContext)
            val state = loaded.state.value
            if (state.isModelLoaded || state is InferenceEngine.State.Error) {
                loaded.cleanUp()
            }
            loaded.loadModel(modelFile.absolutePath)
            loaded.loadFlowModel(flowModelFile.absolutePath)
            loaded.loadHiftModel(hiftModelFile.absolutePath)
            engine = loaded
            loaded
        }

    private suspend fun ensureQwenTokenizerReady(tokenizerDir: File): QwenTokenizerService =
        engineMutex.withLock {
            qwenTokenizerService?.let { return it }
            val loaded = QwenTokenizerService.load(from = tokenizerDir)
            qwenTokenizerService = loaded
            loaded
        }

    private suspend fun ensurePromptFrontEndEngine(resources: LocalResourceFiles): PromptFrontEndEngine =
        engineMutex.withLock {
            promptFrontEndEngine?.let { return it }
            val loaded = PromptFrontEndEngine(
                speechTokenizerModel = resources.speechTokenizer,
                campPlusModel = resources.campPlus,
                mel128Bin = resources.mel128,
                hannWindow400Bin = resources.hannWindow400,
                hannWindow1920Bin = resources.hannWindow,
                librosaMelFnBin = resources.librosaMelFn,
                fftBasisRealBin = resources.fftBasisReal,
                fftBasisImagBin = resources.fftBasisImag,
                poveyWindow400Bin = resources.poveyWindow400,
                melBanks80x257Bin = resources.melBanks80x257,
            )
            promptFrontEndEngine = loaded
            loaded
        }

    private fun installBundledResourcesIfPresent() {
        val targetModelsDir = File(appContext.filesDir, DIRECTORY_MODELS).also {
            if (it.exists() && !it.isDirectory) it.delete()
            if (!it.exists()) it.mkdirs()
        }
        copyAssetDirectoryIfPresent(assetDir = ASSET_DIR_MODELS, targetDir = targetModelsDir)

        val targetResourcesDir = File(appContext.filesDir, DIRECTORY_LOCAL_RESOURCES).also {
            if (it.exists() && !it.isDirectory) it.delete()
            if (!it.exists()) it.mkdirs()
        }
        copyAssetDirectoryIfPresent(assetDir = ASSET_DIR_LOCAL_RESOURCES, targetDir = targetResourcesDir)
    }

    private fun copyAssetDirectoryIfPresent(assetDir: String, targetDir: File) {
        val entries = try {
            appContext.assets.list(assetDir) ?: emptyArray()
        } catch (_: IOException) {
            return
        }
        if (entries.isEmpty()) return

        if (!targetDir.exists()) {
            targetDir.mkdirs()
        }
        val markerFile = File(targetDir, ASSET_SYNC_MARKER)
        if (markerFile.exists()) {
            return
        }

        entries.forEach { entry ->
            val assetPath = "$assetDir/$entry"
            val childEntries = try {
                appContext.assets.list(assetPath) ?: emptyArray()
            } catch (_: IOException) {
                emptyArray()
            }

            if (childEntries.isNotEmpty()) {
                copyAssetDirectoryIfPresent(
                    assetDir = assetPath,
                    targetDir = File(targetDir, entry),
                )
            } else {
                val targetFile = File(targetDir, entry)
                val tempFile = File(targetDir, "${entry}.tmp")
                appContext.assets.open(assetPath).use { input ->
                    tempFile.outputStream().use { output ->
                        input.copyTo(output)
                    }
                }
                if (targetFile.exists()) {
                    targetFile.delete()
                }
                tempFile.renameTo(targetFile)
            }
        }
        markerFile.writeText("ok")
    }

    private fun resolveModelFile(): File {
        val modelsDir = File(appContext.filesDir, DIRECTORY_MODELS).also {
            if (it.exists() && !it.isDirectory) it.delete()
            if (!it.exists()) it.mkdirs()
        }
        return modelsDir.listFiles()
            ?.filter { it.isFile && it.extension.equals("gguf", ignoreCase = true) }
            ?.sortedBy { it.name.lowercase(Locale.ROOT) }
            ?.firstOrNull()
            ?: error("未找到 GGUF 模型，请放入 ${modelsDir.absolutePath}")
    }

    private suspend fun ensureLocalResourcesReady(): LocalResourceFiles =
        engineMutex.withLock {
            localResources?.let { return it }

            installBundledResourcesIfPresent()
            val modelsDir = ensureDirectory(File(appContext.filesDir, DIRECTORY_MODELS))
            val resourcesDir = ensureDirectory(File(appContext.filesDir, DIRECTORY_LOCAL_RESOURCES))

            val resolved = LocalResourceFiles(
                llmModel = resolveOptionalFile(listOf(File(modelsDir, FILE_LLM_MODEL))) ?: resolveModelFile(),
                flowModel = resolveRequiredFile(FILE_FLOW_MODEL, listOf(File(modelsDir, FILE_FLOW_MODEL))),
                hiftModel = resolveRequiredFile(FILE_HIFT_MODEL, listOf(File(modelsDir, FILE_HIFT_MODEL))),
                modelInputZh = resolveRequiredFile(FILE_MODEL_INPUT_ZH, listOf(File(resourcesDir, FILE_MODEL_INPUT_ZH))),
                modelInputZh2 = resolveRequiredFile(FILE_MODEL_INPUT_ZH2, listOf(File(resourcesDir, FILE_MODEL_INPUT_ZH2))),
                modelInputEn = resolveRequiredFile(FILE_MODEL_INPUT_EN, listOf(File(resourcesDir, FILE_MODEL_INPUT_EN))),
                modelInputEn2 = resolveRequiredFile(FILE_MODEL_INPUT_EN2, listOf(File(resourcesDir, FILE_MODEL_INPUT_EN2))),
                modelInputJa = resolveRequiredFile(FILE_MODEL_INPUT_JA, listOf(File(resourcesDir, FILE_MODEL_INPUT_JA))),
                modelInputJa2 = resolveRequiredFile(FILE_MODEL_INPUT_JA2, listOf(File(resourcesDir, FILE_MODEL_INPUT_JA2))),
                llmDecoderWeight = resolveRequiredFile(FILE_LLM_DECODER_WEIGHT, listOf(File(resourcesDir, FILE_LLM_DECODER_WEIGHT))),
                llmDecoderBias = resolveRequiredFile(FILE_LLM_DECODER_BIAS, listOf(File(resourcesDir, FILE_LLM_DECODER_BIAS))),
                speechEmbeddingWeight = resolveRequiredFile(FILE_SPEECH_EMBEDDING_WEIGHT, listOf(File(resourcesDir, FILE_SPEECH_EMBEDDING_WEIGHT))),
                llmEmbedTokens = resolveRequiredFile(FILE_LLM_EMBED_TOKENS, listOf(File(resourcesDir, FILE_LLM_EMBED_TOKENS))),
                noisePe = resolveRequiredFile(FILE_NOISE_PE, listOf(File(resourcesDir, FILE_NOISE_PE))),
                sosEosEmb = resolveRequiredFile(FILE_SOS_EOS_EMB, listOf(File(resourcesDir, FILE_SOS_EOS_EMB))),
                taskIdEmb = resolveRequiredFile(FILE_TASK_ID_EMB, listOf(File(resourcesDir, FILE_TASK_ID_EMB))),
                campPlus = resolveRequiredFile(FILE_CAMPPLUS, listOf(File(resourcesDir, FILE_CAMPPLUS))),
                speechTokenizer = resolveRequiredFile(FILE_SPEECH_TOKENIZER, listOf(File(resourcesDir, FILE_SPEECH_TOKENIZER))),
                mel128 = resolveRequiredFile(FILE_MEL_128, listOf(File(resourcesDir, FILE_MEL_128))),
                librosaMelFn = resolveRequiredFile(FILE_LIBROSA_MEL_FN, listOf(File(resourcesDir, FILE_LIBROSA_MEL_FN))),
                hannWindow = resolveRequiredFile(FILE_HANN_WINDOW, listOf(File(resourcesDir, FILE_HANN_WINDOW))),
                hannWindow400 = resolveRequiredFile(FILE_HANN_WINDOW_400, listOf(File(resourcesDir, FILE_HANN_WINDOW_400))),
                fftBasisReal = resolveRequiredFile(FILE_FFT_BASIS_REAL, listOf(File(resourcesDir, FILE_FFT_BASIS_REAL))),
                fftBasisImag = resolveRequiredFile(FILE_FFT_BASIS_IMAG, listOf(File(resourcesDir, FILE_FFT_BASIS_IMAG))),
                poveyWindow400 = resolveRequiredFile(FILE_POVEY_WINDOW_400, listOf(File(resourcesDir, FILE_POVEY_WINDOW_400))),
                melBanks80x257 = resolveRequiredFile(FILE_MEL_BANKS_80x257, listOf(File(resourcesDir, FILE_MEL_BANKS_80x257))),
                tokenizerDir = resolveRequiredDirectory(DIR_QWEN2_TOKENIZER, listOf(File(resourcesDir, DIR_QWEN2_TOKENIZER))),
            )
            localResources = resolved
            resolved
        }

    private fun ensureDirectory(path: File): File {
        if (path.exists() && !path.isDirectory) {
            error("路径存在但不是目录: ${path.absolutePath}")
        }
        if (!path.exists() && !path.mkdirs()) {
            error("无法创建目录: ${path.absolutePath}")
        }
        return path
    }

    private fun resolveRequiredFile(label: String, candidates: List<File>): File =
        candidates.firstOrNull { it.exists() && it.isFile }
            ?: error(
                buildString {
                    append("缺少文件 ").append(label).append("，请放到以下路径之一：")
                    candidates.forEach { append("\n- ").append(it.absolutePath) }
                },
            )

    private fun resolveRequiredDirectory(label: String, candidates: List<File>): File =
        candidates.firstOrNull { it.exists() && it.isDirectory }
            ?: error(
                buildString {
                    append("缺少目录 ").append(label).append("，请放到以下路径之一：")
                    candidates.forEach { append("\n- ").append(it.absolutePath) }
                },
            )

    private fun resolveOptionalFile(candidates: List<File>): File? =
        candidates.firstOrNull { it.exists() && it.isFile }

    private fun nowSeconds() = System.nanoTime() / 1_000_000_000.0

    companion object {
        private const val ASSET_SYNC_MARKER = ".asset_sync_ok"
        private const val DIRECTORY_MODELS = "models"
        private const val DIRECTORY_LOCAL_RESOURCES = "local_llm_resources"
        private const val ASSET_DIR_MODELS = "models"
        private const val ASSET_DIR_LOCAL_RESOURCES = "local_llm_resources"

        private const val FILE_LLM_MODEL = "cosyvoice2-0.5B-Q2_K.gguf"
        private const val FILE_FLOW_MODEL = "flow_fp32.gguf"
        private const val FILE_HIFT_MODEL = "hift_fp32.gguf"

        private const val FILE_MODEL_INPUT_ZH = "model_input_zh.bin"
        private const val FILE_MODEL_INPUT_ZH2 = "model_input_zh2.bin"
        private const val FILE_MODEL_INPUT_EN = "model_input_en.bin"
        private const val FILE_MODEL_INPUT_EN2 = "model_input_en2.bin"
        private const val FILE_MODEL_INPUT_JA = "model_input_ja.bin"
        private const val FILE_MODEL_INPUT_JA2 = "model_input_ja2.bin"
        private const val FILE_LLM_DECODER_WEIGHT = "llm_decoder_weight.bin"
        private const val FILE_LLM_DECODER_BIAS = "llm_decoder_bias.bin"
        private const val FILE_SPEECH_EMBEDDING_WEIGHT = "speech_embedding_weight.bin"
        private const val FILE_LLM_EMBED_TOKENS = "llm_embed_tokens.bin"
        private const val FILE_NOISE_PE = "noise_pe.bin"
        private const val FILE_SOS_EOS_EMB = "sos_eos_emb.bin"
        private const val FILE_TASK_ID_EMB = "task_id_emb.bin"
        private const val FILE_CAMPPLUS = "campplus.onnx"
        private const val FILE_SPEECH_TOKENIZER = "speech_tokenizer_v2.onnx"
        private const val FILE_MEL_128 = "mel_128.bin"
        private const val FILE_LIBROSA_MEL_FN = "librosa_mel_fn.bin"
        private const val FILE_HANN_WINDOW = "hann_window.bin"
        private const val FILE_HANN_WINDOW_400 = "hann_window_400.bin"
        private const val FILE_FFT_BASIS_REAL = "fft_basis_real.bin"
        private const val FILE_FFT_BASIS_IMAG = "fft_basis_imag.bin"
        private const val FILE_POVEY_WINDOW_400 = "povey_window_400.bin"
        private const val FILE_MEL_BANKS_80x257 = "mel_banks_80x257.bin"
        private const val DIR_QWEN2_TOKENIZER = "qwen2_tokenizer"

        private const val DEFAULT_SYSTEM_PROMPT =
            "You are a concise assistant. Return plain text suitable for speech output."
        private const val LLM_HIDDEN_SIZE = 896
        private const val HIFT_N_FFT = 16
        private const val HIFT_HOP_LENGTH = 4
        private const val HIFT_N_FREQ = HIFT_N_FFT / 2 + 1
        private const val HIFT_FRAME_FEATURES = HIFT_N_FREQ * 2
    }
}

private data class FrontEndResult(
    val speechTokenLen: Int,
    val speechFeatLen: Int,
    val speechEmbeddingLen: Int,
    val speechTokens: IntArray,
    val speechFeat: FloatArray,
    val speechEmbedding: FloatArray,
)

private data class LLMResult(
    val tokenCount: Int,
    val speechTokens: IntArray,
)

private data class FlowResult(
    val units: Int,
    val output: FloatArray,
)

private data class HiftResult(
    val outputSamples: Int,
    val output: FloatArray,
)

private data class NoisePeData(
    val randNoise: FloatArray,
    val extendPe: FloatArray,
)

private data class LocalResourceFiles(
    val llmModel: File,
    val flowModel: File,
    val hiftModel: File,
    val modelInputZh: File,
    val modelInputZh2: File,
    val modelInputEn: File,
    val modelInputEn2: File,
    val modelInputJa: File,
    val modelInputJa2: File,
    val llmDecoderWeight: File,
    val llmDecoderBias: File,
    val speechEmbeddingWeight: File,
    val llmEmbedTokens: File,
    val noisePe: File,
    val sosEosEmb: File,
    val taskIdEmb: File,
    val campPlus: File,
    val speechTokenizer: File,
    val mel128: File,
    val librosaMelFn: File,
    val hannWindow: File,
    val hannWindow400: File,
    val fftBasisReal: File,
    val fftBasisImag: File,
    val poveyWindow400: File,
    val melBanks80x257: File,
    val tokenizerDir: File,
)
