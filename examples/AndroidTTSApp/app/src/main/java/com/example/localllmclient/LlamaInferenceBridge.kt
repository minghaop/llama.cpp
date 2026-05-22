package com.example.llama

import android.content.Context
import android.os.Trace
import android.util.Log
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
import java.util.zip.ZipFile
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
    data class FlowBreakdown(val encoderSeconds: Double, val decoderSeconds: Double, val totalSeconds: Double) : InferenceEvent()
    data class Note(val message: String) : InferenceEvent()
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
    @Volatile
    private var onnxQnnFlowRunner: OnnxQnnFlowRunner? = null
    @Volatile
    private var liteRtFlowRunner: LiteRtFlowRunner? = null
    @Volatile
    private var liteRtNativeFlowRunner: LiteRtNativeFlowRunner? = null
    @Volatile
    private var mnnFlowEncoderRunner: MnnFlowRunner? = null
    @Volatile
    private var mnnFlowDecoderRunner: MnnFlowDecoderRunner? = null
    @Volatile
    private var mnnHifiGanRunner: MnnHifiGanRunner? = null
    @Volatile
    private var mnnLlmRunner: MnnLlmRunner? = null

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
        val runStart = nowSeconds()
        fun done(result: String, stage: InferenceStage): String {
            Log.i(
                TAG,
                "runInference finished: stage=$stage result=$result total=${"%.3f".format(Locale.US, nowSeconds() - runStart)} s",
            )
            return result
        }

        var currentStage = InferenceStage.frontEnd
        Trace.beginSection("runInference_total")
        return try {
            Log.i(TAG, "runInference started")
            if (LLM_ONLY_TEST_MODE) {
                currentStage = InferenceStage.llm
                val resources = ensureLocalResourcesReady()
                val llmRunner = ensureMnnLlmRunnerReady(resources)
                val llmOutput = runLLMInferenceFromPt(
                    runner = llmRunner,
                    resources = resources,
                    lmInputPtFile = resources.lmInputPt,
                    onEvent = onEvent,
                )
                return done("LLM_ONLY_OK:promptSeq=${llmOutput.promptSeqLen},newTokens=${llmOutput.generatedTokens}", currentStage)
            }
            if (HIFIGAN_ONLY_TEST_MODE) {
                currentStage = InferenceStage.hift
                emit(InferenceEvent.StageBegan(InferenceStage.hift, "HifiGan init: 正在加载模型"))
                val resources = ensureHifiGanOnlyResourcesReady()
                Log.i(TAG, "[HifiGanOnly] enabled, skip FrontEnd/LLM/Flow/VoiceGeneration and run hifigan with prepacked inputs")
                val hifiganRunner = ensureMnnHifiGanRunnerReady(resources.hifiganModel)
                emit(
                    InferenceEvent.StageProgress(
                        stage = InferenceStage.hift,
                        unitName = "HifiGan init: 模型加载完成",
                        unitsDone = 1,
                        secondsElapsed = 0.0,
                        instUPS = 0.0,
                        avgUPS = 0.0,
                    ),
                )
                val hiftResult = runHifiGanInferenceFromBins(
                    runner = hifiganRunner,
                    inputDir = resources.hifiganInputsBinDir,
                    onEvent = onEvent,
                )
                return done("HIFIGAN_ONLY_OK:units=${hiftResult.outputSamples}", currentStage)
            }

            if (FLOW_ONLY_TEST_MODE) {
                currentStage = InferenceStage.flow
                val resources = ensureFlowOnlyResourcesReady()
                val flowResult = when (FLOW_ONLY_BACKEND) {
                    FlowOnlyBackend.LITERT_CPP -> {
                        Log.i(TAG, "[FlowOnly][LiteRT][C++] enabled, run flow.tflite with prepacked bins")
                        Log.i(TAG, "[FlowOnly][LiteRT][C++] flowModel=${resources.flowLiteRtModel.absolutePath}")
                        emit(InferenceEvent.StageBegan(InferenceStage.flow, "Flow init: loading LiteRT C++ flow model"))
                        val flowRunner = ensureLiteRtNativeFlowRunnerReady(flowLiteRtModelFile = resources.flowLiteRtModel)
                        emit(InferenceEvent.Note("[FlowOnly][LiteRT][C++] runtime=${flowRunner.runtimeMode}"))
                        emit(
                            InferenceEvent.StageProgress(
                                stage = InferenceStage.flow,
                                unitName = "Flow init: LiteRT C++ model ready",
                                unitsDone = 1,
                                secondsElapsed = 0.0,
                                instUPS = 0.0,
                                avgUPS = 0.0,
                            ),
                        )
                        runFlowInferenceFromBins(
                            runner = flowRunner,
                            inputDir = resources.flowInputsBinDir,
                            onEvent = onEvent,
                        )
                    }
                    FlowOnlyBackend.LITERT -> {
                        Log.i(TAG, "[FlowOnly][LiteRT] enabled, run flow.tflite with prepacked bins")
                        Log.i(TAG, "[FlowOnly][LiteRT] flowModel=${resources.flowLiteRtModel.absolutePath}")
                        emit(InferenceEvent.StageBegan(InferenceStage.flow, "Flow init: loading LiteRT flow model"))
                        val flowRunner = ensureLiteRtFlowRunnerReady(flowLiteRtModelFile = resources.flowLiteRtModel)
                        emit(InferenceEvent.Note("[FlowOnly][LiteRT] runtime=${flowRunner.runtimeMode}"))
                        emit(
                            InferenceEvent.StageProgress(
                                stage = InferenceStage.flow,
                                unitName = "Flow init: LiteRT model ready",
                                unitsDone = 1,
                                secondsElapsed = 0.0,
                                instUPS = 0.0,
                                avgUPS = 0.0,
                            ),
                        )
                        runFlowInferenceFromBins(
                            runner = flowRunner,
                            inputDir = resources.flowInputsBinDir,
                            onEvent = onEvent,
                        )
                    }
                    FlowOnlyBackend.LLAMA_CPP -> {
                        Log.i(TAG, "[FlowOnly][llama.cpp] enabled, skip FrontEnd/LLM/HIFT and run flow model with prepacked bins")
                        Log.i(TAG, "[FlowOnly][llama.cpp] flowModel=${resources.flowModel.absolutePath}")
                        emit(InferenceEvent.StageBegan(InferenceStage.flow, "Flow init: loading llama.cpp flow model"))
                        val flowEngine = ensureFlowOnlyEngineReady(flowModelFile = resources.flowModel)
                        emit(
                            InferenceEvent.StageProgress(
                                stage = InferenceStage.flow,
                                unitName = "Flow init: llama.cpp flow model ready",
                                unitsDone = 1,
                                secondsElapsed = 0.0,
                                instUPS = 0.0,
                                avgUPS = 0.0,
                            ),
                        )
                        runFlowInferenceFromBins(
                            engine = flowEngine,
                            inputDir = resources.flowInputsBinDir,
                            noisePeFile = resources.noisePe,
                            onEvent = onEvent,
                        )
                    }
                    FlowOnlyBackend.ONNX_QNN_GPU -> {
                        emit(InferenceEvent.StageBegan(InferenceStage.flow, "Flow init: loading ONNX QNN GPU"))
                        val flowRunner = ensureOnnxQnnFlowRunnerReady(flowQnnModelFile = resources.flowQnnModel)
                        emit(
                            InferenceEvent.StageProgress(
                                stage = InferenceStage.flow,
                                unitName = "Flow init: ONNX QNN GPU ready",
                                unitsDone = 1,
                                secondsElapsed = 0.0,
                                instUPS = 0.0,
                                avgUPS = 0.0,
                            ),
                        )
                        runFlowInferenceFromBins(
                            runner = flowRunner,
                            inputDir = resources.flowInputsBinDir,
                            onEvent = onEvent,
                        )
                    }
                    FlowOnlyBackend.MNN_ENCODER_DECODER -> {
                        emit(InferenceEvent.StageBegan(InferenceStage.flow, "Encoder/Decoder init: loading MNN models"))
                        val flowEncoderRunner = ensureMnnFlowEncoderRunnerReady(flowEncoderModelFile = resources.flowEncoderModel)
                        val flowDecoderRunner = ensureMnnFlowDecoderRunnerReady(flowDecoderModelFile = resources.flowDecoderModel)
                        emit(
                            InferenceEvent.StageProgress(
                                stage = InferenceStage.flow,
                                unitName = "Encoder/Decoder init: MNN models loaded",
                                unitsDone = 1,
                                secondsElapsed = 0.0,
                                instUPS = 0.0,
                                avgUPS = 0.0,
                            ),
                        )
                        runFlowInferenceFromBins(
                            encoderRunner = flowEncoderRunner,
                            decoderRunner = flowDecoderRunner,
                            inputDir = resources.flowInputsBinDir,
                            onEvent = onEvent,
                        )
                    }
                }
                val tag = when (FLOW_ONLY_BACKEND) {
                    FlowOnlyBackend.LITERT_CPP -> "FLOW_ONLY_LITERT_CPP_OK"
                    FlowOnlyBackend.LITERT -> "FLOW_ONLY_LITERT_OK"
                    FlowOnlyBackend.LLAMA_CPP -> "FLOW_ONLY_LLAMA_CPP_OK"
                    FlowOnlyBackend.ONNX_QNN_GPU -> "FLOW_ONLY_ONNX_QNN_OK"
                    FlowOnlyBackend.MNN_ENCODER_DECODER -> "ENCODER_DECODER_ONLY_OK"
                }
                return done("$tag:units=${flowResult.units}", currentStage)
            }

            val resources = ensureLocalResourcesReady()

            currentStage = InferenceStage.frontEnd
            val frontEndResult = runFrontEndInference(
                promptAudio = promptAudio,
                promptSampleRate = promptSampleRate,
                resources = resources,
                onEvent = onEvent,
            )

            currentStage = InferenceStage.llmPrepare
            Log.i(
                TAG,
                "[LLMPrepare] begin ensureEngineReady, llm=${resources.llmModel.absolutePath}, " +
                    "flow=${resources.flowModel.absolutePath}, hifigan=${resources.hifiganModel.absolutePath}",
            )
            val loadedEngine = if (USE_MNN_LLM_BACKEND) {
                ensureFlowOnlyEngineReady(flowModelFile = resources.flowModel)
            } else {
                ensureEngineReady(
                    modelFile = resources.llmModel,
                    flowModelFile = resources.flowModel,
                )
            }
            Log.i(TAG, "[LLMPrepare] ensureEngineReady done")
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
            emit(InferenceEvent.StageBegan(InferenceStage.hift, "HifiGan init: 正在加载模型"))
            val hifiganRunner = ensureMnnHifiGanRunnerReady(resources.hifiganModel)
            emit(
                InferenceEvent.StageProgress(
                    stage = InferenceStage.hift,
                    unitName = "HifiGan init: 模型加载完成",
                    unitsDone = 1,
                    secondsElapsed = 0.0,
                    instUPS = 0.0,
                    avgUPS = 0.0,
                ),
            )
            val hiftResult = runHIFTInference(
                flowResult = flowResult,
                runner = hifiganRunner,
                onEvent = onEvent,
            )

            currentStage = InferenceStage.voiceGeneration
            done(runVoiceGeneration(
                hiftResult = hiftResult,
                onEvent = onEvent,
            ), currentStage)
        } catch (t: Throwable) {
            Log.e(TAG, "runInference failed at $currentStage", t)
            emit(InferenceEvent.Failed(currentStage, t.message ?: t.toString()))
            done("false", currentStage)
        } finally {
            Trace.endSection()
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
        val activeMnnFlowEncoder = mnnFlowEncoderRunner
        if (activeMnnFlowEncoder != null) {
            runCatching { activeMnnFlowEncoder.close() }
            mnnFlowEncoderRunner = null
        }
        val activeMnnFlowDecoder = mnnFlowDecoderRunner
        if (activeMnnFlowDecoder != null) {
            runCatching { activeMnnFlowDecoder.close() }
            mnnFlowDecoderRunner = null
        }
        val activeOnnxQnnFlow = onnxQnnFlowRunner
        if (activeOnnxQnnFlow != null) {
            runCatching { activeOnnxQnnFlow.close() }
            onnxQnnFlowRunner = null
        }
        val activeLiteRtFlow = liteRtFlowRunner
        if (activeLiteRtFlow != null) {
            runCatching { activeLiteRtFlow.close() }
            liteRtFlowRunner = null
        }
        val activeLiteRtNativeFlow = liteRtNativeFlowRunner
        if (activeLiteRtNativeFlow != null) {
            runCatching { activeLiteRtNativeFlow.close() }
            liteRtNativeFlowRunner = null
        }
        val activeMnnHifiGan = mnnHifiGanRunner
        if (activeMnnHifiGan != null) {
            runCatching { activeMnnHifiGan.close() }
            mnnHifiGanRunner = null
        }
        val activeMnnLlm = mnnLlmRunner
        if (activeMnnLlm != null) {
            runCatching { activeMnnLlm.close() }
            mnnLlmRunner = null
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
        Log.i(TAG, "[LLMPrepare] tokenizer init begin: ${resources.tokenizerDir.absolutePath}")

        val tokenizerEntries = resources.tokenizerDir.list()?.size ?: 0
        val tokenizer = ensureQwenTokenizerReady(resources.tokenizerDir)
        Log.i(TAG, "[LLMPrepare] tokenizer init done, entries=$tokenizerEntries")
        Log.i(TAG, "[LLMPrepare] tokenizer encode begin")
        val ttsTokenIds = tokenizer.encodeSync(ttsText)
        val promptTokenIds = tokenizer.encodeSync(promptText)
        Log.i(
            TAG,
            "[LLMPrepare] tokenizer encode done, ttsLen=${ttsTokenIds.size}, promptLen=${promptTokenIds.size}",
        )
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
        Log.i(TAG, "[LLMPrepare] done in ${"%.3f".format(Locale.US, nowSeconds() - prepBegin)} s")

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

        val mnnRunner = if (USE_MNN_LLM_BACKEND) {
            ensureMnnLlmRunnerReady(resources)
        } else {
            null
        }
        if (USE_MNN_LLM_BACKEND) {
            mnnRunner!!.resetKvCache()
        } else {
            engine.resetKvCache()
        }

        val reportEverySeconds = 1.0
        var lastReportT = llmBegin
        var lastReportCount = 0

        for (i in 0 until maxLen) {
            val llmRes = if (USE_MNN_LLM_BACKEND) {
                mnnRunner!!.decodeEmbeddings(currentInput, nPast)
            } else {
                engine.decodeEmbeddings(currentInput, nPast)
            }
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
            Log.i(
                TAG,
                "[LLM] step=$i topId=$topId nPast=$nPast generated=${outTokens.size}",
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

    private suspend fun runFlowInferenceFromBins(
        engine: InferenceEngine,
        inputDir: File,
        noisePeFile: File,
        onEvent: ((InferenceEvent) -> Unit)? = null,
    ): FlowResult {
        fun emit(event: InferenceEvent) {
            onEvent?.invoke(event)
        }

        val t0 = nowSeconds()
        emit(InferenceEvent.StageBegan(InferenceStage.flow, "Flow inference (prepacked bins, llama.cpp)"))
        Trace.beginSection("FlowOnly_llamaCpp_total")
        try {

            val tLoadInput0 = nowSeconds()
            Trace.beginSection("FlowOnly_loadInput")
            val inputs = try {
                readFlowOnlyInputBundle(inputDir)
            } finally {
                Trace.endSection()
            }
            Trace.beginSection("FlowOnly_loadNoisePe")
            val noise = try {
                loadNoisePe(noisePeFile)
            } finally {
                Trace.endSection()
            }
            val loadInputSeconds = nowSeconds() - tLoadInput0
            val flowTokenCount = inputs.tokenLen.coerceAtMost(inputs.flowToken.size)
            val promptTokenCount = inputs.promptTokenLen.coerceAtMost(inputs.promptToken.size)
            require(flowTokenCount > 0) { "flow token count must be > 0, got=$flowTokenCount" }
            require(promptTokenCount > 0) { "prompt token count must be > 0, got=$promptTokenCount" }
            val tMergeToken0 = nowSeconds()
            Trace.beginSection("FlowOnly_mergeToken")
            val mergedTokens = try {
                val tmp = IntArray(flowTokenCount + promptTokenCount)
                var i = 0
                while (i < promptTokenCount) {
                    tmp[i] = inputs.promptToken[i].toInt()
                    i += 1
                }
                var j = 0
                while (j < flowTokenCount) {
                    tmp[promptTokenCount + j] = inputs.flowToken[j].toInt()
                    j += 1
                }
                tmp
            } finally {
                Trace.endSection()
            }
            val mergeTokenSeconds = nowSeconds() - tMergeToken0
        Log.i(
            TAG,
            "[FlowOnly][llama.cpp] inputs loaded, token=${inputs.flowToken.size}, tokenLen=${inputs.tokenLen}, " +
                "promptToken=${inputs.promptToken.size}, promptTokenLen=${inputs.promptTokenLen}, " +
                "promptFeat=${inputs.promptFeat.size}, promptFeatLen=${inputs.promptFeatLen}, " +
                "embedding=${inputs.embedding.size}, mergedToken=${mergedTokens.size}",
        )

            val tNative0 = nowSeconds()
            Trace.beginSection("FlowOnly_nativeEncodeFlow")
            val flowOutput = try {
                engine.encodeFlow(
                    inputEmbeddings = inputs.embedding,
                    flowFeat = inputs.promptFeat,
                    flowToken = mergedTokens,
                    tokenLen = flowTokenCount,
                    promptTokenLen = promptTokenCount,
                    promptFeatLen = inputs.promptFeatLen,
                    randNoise = noise.randNoise,
                    extendPe = noise.extendPe,
                )
            } finally {
                Trace.endSection()
            }
            val nativeSeconds = nowSeconds() - tNative0

            val tPost0 = nowSeconds()
            Trace.beginSection("FlowOnly_post")
            try {
                emit(InferenceEvent.Note(buildDecoderHeadPreviewNote(flowOutput)))
            } finally {
                Trace.endSection()
            }
            val postSeconds = nowSeconds() - tPost0
            val units = inputs.flowToken.size.coerceAtLeast(1)
            val seconds = nowSeconds() - t0
            Log.i(
                TAG,
                "[FlowOnly][llama.cpp] timing: total=${"%.3f".format(Locale.US, seconds)}s, " +
                    "loadInput=${"%.3f".format(Locale.US, loadInputSeconds)}s, " +
                    "mergeToken=${"%.3f".format(Locale.US, mergeTokenSeconds)}s, " +
                    "nativeEncodeFlow=${"%.3f".format(Locale.US, nativeSeconds)}s, " +
                    "post=${"%.3f".format(Locale.US, postSeconds)}s",
            )
            emit(
                InferenceEvent.Note(
                    "[FlowOnly][llama.cpp] total=${"%.3f".format(Locale.US, seconds)}s " +
                        "(loadInput=${"%.3f".format(Locale.US, loadInputSeconds)}s, " +
                        "mergeToken=${"%.3f".format(Locale.US, mergeTokenSeconds)}s, " +
                        "nativeEncodeFlow=${"%.3f".format(Locale.US, nativeSeconds)}s, " +
                        "post=${"%.3f".format(Locale.US, postSeconds)}s)",
                ),
            )
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
        } finally {
            Trace.endSection()
        }
    }

    private suspend fun runFlowInferenceFromBins(
        encoderRunner: MnnFlowRunner,
        decoderRunner: MnnFlowDecoderRunner,
        inputDir: File,
        onEvent: ((InferenceEvent) -> Unit)? = null,
    ): FlowResult {
        fun emit(event: InferenceEvent) {
            onEvent?.invoke(event)
        }

        val t0 = nowSeconds()
        emit(InferenceEvent.StageBegan(InferenceStage.flow, "Encoder/Decoder inference (prepacked inputs)"))

        val inputs = readFlowOnlyInputBundle(inputDir)
        val decoderInputs = readFlowDecoderInputPt(File(inputDir, FILE_FLOW_DECODER_INPUT_PT))
        Log.i(
            TAG,
            "[EncDecOnly][MNN] encoder inputs loaded, token=${inputs.flowToken.size}, tokenLen=${inputs.tokenLen}, " +
                "promptToken=${inputs.promptToken.size}, promptTokenLen=${inputs.promptTokenLen}, " +
                "promptFeat=${inputs.promptFeat.size}, promptFeatLen=${inputs.promptFeatLen}, " +
                "embedding=${inputs.embedding.size}, streaming=${inputs.streaming}, finalize=${inputs.finalize}",
        )
        Log.i(TAG, decoderInputs.describeForLog())

        val encoderStart = nowSeconds()
        encoderRunner.forward(
            token = inputs.flowToken,
            tokenLen = inputs.tokenLen,
            promptToken = inputs.promptToken,
            promptTokenLen = inputs.promptTokenLen,
            promptFeat = inputs.promptFeat,
            promptFeatLen = inputs.promptFeatLen,
            embedding = inputs.embedding,
            streaming = inputs.streaming,
            finalize = inputs.finalize,
        )
        val encoderSeconds = nowSeconds() - encoderStart
        val decoderStart = nowSeconds()
        val flowOutput = when (decoderInputs.mode) {
            FlowDecoderInputMode.Legacy5 -> decoderRunner.forward(
                mu = decoderInputs.mu,
                mask = decoderInputs.mask,
                z = decoderInputs.z,
                spks = decoderInputs.spks,
                cond = decoderInputs.cond,
            )
            FlowDecoderInputMode.Prepared6 -> decoderRunner.forwardPrepared6(
                xIn = decoderInputs.xIn,
                maskIn = decoderInputs.maskIn,
                muIn = decoderInputs.muIn,
                tIn = decoderInputs.tIn,
                spksIn = decoderInputs.spksIn,
                condIn = decoderInputs.condIn,
            )
        }
        val decoderSeconds = nowSeconds() - decoderStart
        val totalSeconds = nowSeconds() - t0

        val opProfileSummary = encoderRunner.consumeLastProfileSummary()
        if (!opProfileSummary.isNullOrBlank()) {
            emit(InferenceEvent.Note("[EncoderOp] $opProfileSummary"))
        }
        val decoderProfileSummary = decoderRunner.consumeLastProfileSummary()
        if (!decoderProfileSummary.isNullOrBlank()) {
            emit(InferenceEvent.Note("[DecoderOp] $decoderProfileSummary"))
        }
        emit(InferenceEvent.Note(buildDecoderHeadPreviewNote(flowOutput)))
        emit(
            InferenceEvent.FlowBreakdown(
                encoderSeconds = encoderSeconds,
                decoderSeconds = decoderSeconds,
                totalSeconds = totalSeconds,
            ),
        )

        val units = inputs.flowToken.size.coerceAtLeast(1)
        emit(
            InferenceEvent.StageEnded(
                StageEndedInfo(
                    stage = InferenceStage.flow,
                    unitName = "Encoder/Decoder inference completed",
                    units = units,
                    seconds = totalSeconds,
                    avgUPS = if (totalSeconds > 0) units / totalSeconds else 0.0,
                ),
            ),
        )
        return FlowResult(
            units = units,
            output = flowOutput,
        )
    }

    private suspend fun runFlowInferenceFromBins(
        runner: OnnxQnnFlowRunner,
        inputDir: File,
        onEvent: ((InferenceEvent) -> Unit)? = null,
    ): FlowResult {
        fun emit(event: InferenceEvent) {
            onEvent?.invoke(event)
        }

        val t0 = nowSeconds()
        emit(InferenceEvent.StageBegan(InferenceStage.flow, "Flow inference (prepacked inputs, ONNX QNN GPU)"))

        val inputs = readFlowOnlyInputBundle(inputDir)
        Log.i(
            TAG,
            "[FlowOnly][ONNX-QNN] inputs loaded, token=${inputs.flowToken.size}, tokenLen=${inputs.tokenLen}, " +
                "promptToken=${inputs.promptToken.size}, promptTokenLen=${inputs.promptTokenLen}, " +
                "promptFeat=${inputs.promptFeat.size}, promptFeatLen=${inputs.promptFeatLen}, " +
                "embedding=${inputs.embedding.size}, streaming=${inputs.streaming}, finalize=${inputs.finalize}",
        )
        emit(InferenceEvent.Note("[FlowOnly] ONNX Runtime QNN backend=$DEFAULT_QNN_GPU_BACKEND_PATH"))
        if (!runner.cpuFallbackDisabled) {
            emit(InferenceEvent.Note("[FlowOnly] ONNX QNN is running with CPU fallback enabled for unsupported ops"))
        }

        val flowOutput = runner.forward(
            token = inputs.flowToken,
            tokenLen = inputs.tokenLen,
            promptToken = inputs.promptToken,
            promptTokenLen = inputs.promptTokenLen,
            promptFeat = inputs.promptFeat,
            promptFeatLen = inputs.promptFeatLen,
            embedding = inputs.embedding,
            streaming = inputs.streaming,
            finalize = inputs.finalize,
        )
        emit(InferenceEvent.Note(buildDecoderHeadPreviewNote(flowOutput)))

        val units = inputs.flowToken.size.coerceAtLeast(1)
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
        return FlowResult(
            units = units,
            output = flowOutput,
        )
    }

    private suspend fun runFlowInferenceFromBins(
        runner: LiteRtFlowRunner,
        inputDir: File,
        onEvent: ((InferenceEvent) -> Unit)? = null,
    ): FlowResult {
        fun emit(event: InferenceEvent) {
            onEvent?.invoke(event)
        }

        val t0 = nowSeconds()
        emit(InferenceEvent.StageBegan(InferenceStage.flow, "Flow inference (prepacked inputs, LiteRT)"))

        val inputs = readFlowOnlyInputBundle(inputDir)
        Log.i(
            TAG,
            "[FlowOnly][LiteRT] inputs loaded, token=${inputs.flowToken.size}, tokenLen=${inputs.tokenLen}, " +
                "promptToken=${inputs.promptToken.size}, promptTokenLen=${inputs.promptTokenLen}, " +
                "promptFeat=${inputs.promptFeat.size}, promptFeatLen=${inputs.promptFeatLen}, " +
                "embedding=${inputs.embedding.size}, streaming=${inputs.streaming}, finalize=${inputs.finalize}",
        )
        emit(InferenceEvent.Note("[FlowOnly][LiteRT] runtime=${runner.runtimeMode}"))
        val streamingForRun = false
        val finalizeForRun = true
        Log.i(
            TAG,
            "[FlowOnly][LiteRT] force streaming=0/finalize=1, rawStreaming=${inputs.streaming}, rawFinalize=${inputs.finalize}",
        )

        val flowOutput = runner.forward(
            token = inputs.flowToken,
            tokenLen = inputs.tokenLen,
            promptToken = inputs.promptToken,
            promptTokenLen = inputs.promptTokenLen,
            promptFeat = inputs.promptFeat,
            promptFeatLen = inputs.promptFeatLen,
            embedding = inputs.embedding,
            streaming = streamingForRun,
            finalize = finalizeForRun,
        )
        val nonZero = flowOutput.count { it != 0.0f }
        var minV = Float.POSITIVE_INFINITY
        var maxV = Float.NEGATIVE_INFINITY
        var sumV = 0.0
        for (v in flowOutput) {
            if (v < minV) minV = v
            if (v > maxV) maxV = v
            sumV += v
        }
        val meanV = if (flowOutput.isNotEmpty()) sumV / flowOutput.size else 0.0
        Log.i(
            TAG,
            "[FlowOnly][LiteRT] output stats: size=${flowOutput.size}, nonZero=$nonZero, " +
                "min=${"%.6f".format(Locale.US, minV)}, max=${"%.6f".format(Locale.US, maxV)}, mean=${"%.6f".format(Locale.US, meanV)}",
        )

        emit(InferenceEvent.Note(buildDecoderHeadPreviewNote(flowOutput)))
        val units = inputs.flowToken.size.coerceAtLeast(1)
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
        return FlowResult(
            units = units,
            output = flowOutput,
        )
    }

    private suspend fun runFlowInferenceFromBins(
        runner: LiteRtNativeFlowRunner,
        inputDir: File,
        onEvent: ((InferenceEvent) -> Unit)? = null,
    ): FlowResult {
        fun emit(event: InferenceEvent) {
            onEvent?.invoke(event)
        }

        val t0 = nowSeconds()
        emit(InferenceEvent.StageBegan(InferenceStage.flow, "Flow inference (prepacked inputs, LiteRT C++)"))
        emit(InferenceEvent.Note("[FlowOnly][LiteRT] runtime=${runner.runtimeMode}"))

        val flowOutput = runner.forwardFromBin(inputDir)
        val nonZero = flowOutput.count { it != 0.0f }
        var minV = Float.POSITIVE_INFINITY
        var maxV = Float.NEGATIVE_INFINITY
        var sumV = 0.0
        for (v in flowOutput) {
            if (v < minV) minV = v
            if (v > maxV) maxV = v
            sumV += v
        }
        val meanV = if (flowOutput.isNotEmpty()) sumV / flowOutput.size else 0.0
        Log.i(
            TAG,
            "[FlowOnly][LiteRT][C++] output stats: size=${flowOutput.size}, nonZero=$nonZero, " +
                "min=${"%.6f".format(Locale.US, minV)}, max=${"%.6f".format(Locale.US, maxV)}, mean=${"%.6f".format(Locale.US, meanV)}",
        )
        emit(InferenceEvent.Note(buildDecoderHeadPreviewNote(flowOutput)))

        val units = (File(inputDir, FILE_FLOW_INPUT_TOKEN).length() / Long.SIZE_BYTES).toInt().coerceAtLeast(1)
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
        return FlowResult(
            units = units,
            output = flowOutput,
        )
    }

    private fun readFlowOnlyInputBundle(inputDir: File): FlowOnlyInputBundle =
        FlowOnlyInputBundle(
            flowToken = readRawInt64Array(File(inputDir, FILE_FLOW_INPUT_TOKEN)),
            tokenLen = readRawInt32Scalar(File(inputDir, FILE_FLOW_INPUT_TOKEN_LEN)),
            promptToken = readRawInt64Array(File(inputDir, FILE_FLOW_INPUT_PROMPT_TOKEN)),
            promptTokenLen = readRawInt32Scalar(File(inputDir, FILE_FLOW_INPUT_PROMPT_TOKEN_LEN)),
            promptFeat = readRawFloatArray(File(inputDir, FILE_FLOW_INPUT_PROMPT_FEAT)),
            promptFeatLen = readRawInt32Scalar(File(inputDir, FILE_FLOW_INPUT_PROMPT_FEAT_LEN)),
            embedding = readRawFloatArray(File(inputDir, FILE_FLOW_INPUT_EMBEDDING)),
            streaming = readRawBoolByte(File(inputDir, FILE_FLOW_INPUT_STREAMING)),
            finalize = readRawBoolByte(File(inputDir, FILE_FLOW_INPUT_FINALIZE)),
        )

    private fun readFlowDecoderInputPt(ptFile: File): FlowDecoderInputBundle {
        require(ptFile.exists() && ptFile.isFile) { "${ptFile.name} not found: ${ptFile.absolutePath}" }
        java.util.zip.ZipFile(ptFile).use { zip ->
            val entries = ArrayList<java.util.zip.ZipEntry>()
            val enumeration = zip.entries()
            while (enumeration.hasMoreElements()) {
                entries.add(enumeration.nextElement())
            }
            val sortedDataEntries = entries
                .filter { !it.isDirectory && it.name.contains("/data/") && it.size > 0L }
                .mapNotNull { entry ->
                    val suffix = entry.name.substringAfterLast("/data/", missingDelimiterValue = "")
                    val index = suffix.toIntOrNull() ?: return@mapNotNull null
                    index to entry
                }
                .sortedBy { it.first }
            require(sortedDataEntries.size >= 5) {
                "${ptFile.name} missing storage entries, expected >=5, got=${sortedDataEntries.size}"
            }
            val storage = sortedDataEntries.associate { (idx, entry) ->
                idx to readFloatStorageEntry(zip, entry)
            }
            val data0 = storage[0] ?: error("${ptFile.name} missing data/0")
            val data1 = storage[1] ?: error("${ptFile.name} missing data/1")
            val data2 = storage[2] ?: error("${ptFile.name} missing data/2")
            val data3 = storage[3] ?: error("${ptFile.name} missing data/3")
            val data4 = storage[4] ?: error("${ptFile.name} missing data/4")

            val data5 = storage[5]
            val looksLikePrepared6 =
                data5 != null &&
                    data0.size == data2.size &&
                    data0.size == data5.size &&
                    data3.size == 2 &&
                    data4.size == FLOW_DECODER_MEL_BINS * 2
            if (looksLikePrepared6) {
                require(data0.size % (FLOW_DECODER_MEL_BINS * 2) == 0) {
                    "flow decoder x_in size invalid: ${data0.size}, not divisible by ${FLOW_DECODER_MEL_BINS * 2}"
                }
                val seqLen = data0.size / (FLOW_DECODER_MEL_BINS * 2)
                require(data1.size == seqLen * 2) {
                    "flow decoder mask_in size invalid: ${data1.size}, expected=${seqLen * 2}"
                }
                return FlowDecoderInputBundle(
                    mode = FlowDecoderInputMode.Prepared6,
                    xIn = data0,
                    maskIn = data1,
                    muIn = data2,
                    tIn = data3,
                    spksIn = data4,
                    condIn = data5,
                )
            }

            val mu = data0
            val mask = data1
            val z = data2
            val spks = data3
            val cond = data4

            require(mu.size % FLOW_DECODER_MEL_BINS == 0) {
                "flow decoder mu size invalid: ${mu.size}, not divisible by $FLOW_DECODER_MEL_BINS"
            }
            require(z.size == mu.size && cond.size == mu.size) {
                "flow decoder z/cond size mismatch: mu=${mu.size}, z=${z.size}, cond=${cond.size}"
            }
            require(spks.size == FLOW_DECODER_MEL_BINS) {
                "flow decoder spks size invalid: ${spks.size}, expected $FLOW_DECODER_MEL_BINS"
            }
            val seqLen = mu.size / FLOW_DECODER_MEL_BINS
            require(mask.size == seqLen) {
                "flow decoder mask size invalid: ${mask.size}, expected seqLen=$seqLen"
            }
            return FlowDecoderInputBundle(
                mode = FlowDecoderInputMode.Legacy5,
                mu = mu,
                mask = mask,
                z = z,
                spks = spks,
                cond = cond,
            )
        }
    }

    private fun readFloatStorageEntry(zip: java.util.zip.ZipFile, entry: java.util.zip.ZipEntry): FloatArray {
        val bytes = zip.getInputStream(entry).use { it.readBytes() }
        require(bytes.size % 4 == 0) {
            "Invalid float storage bytes in ${entry.name}: ${bytes.size}"
        }
        val out = FloatArray(bytes.size / 4)
        val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        for (i in out.indices) {
            out[i] = bb.float
        }
        return out
    }

    private suspend fun runHIFTInference(
        flowResult: FlowResult,
        runner: MnnHifiGanRunner,
        onEvent: ((InferenceEvent) -> Unit)? = null,
    ): HiftResult {
        fun emit(event: InferenceEvent) {
            onEvent?.invoke(event)
        }

        emit(
            InferenceEvent.StageProgress(
                stage = InferenceStage.hift,
                unitName = "HifiGan 推理中...",
                unitsDone = 1,
                secondsElapsed = 0.0,
                instUPS = 0.0,
                avgUPS = 0.0,
            ),
        )

        require(flowResult.output.isNotEmpty()) { "HIFT input is empty" }

        val shape = inferHifiGanInputShape(flowResult.output.size)
        val t0 = nowSeconds()
        val hiftOutput = runner.forward(
            input = flowResult.output,
            shape = shape,
        )
        require(hiftOutput.isNotEmpty()) { "HIFT output is empty" }
        val outputSamples = hiftOutput.size
        val elapsed = nowSeconds() - t0
        emit(
            InferenceEvent.StageEnded(
                StageEndedInfo(
                    stage = InferenceStage.hift,
                    unitName = "HifiGan Inference completed",
                    units = outputSamples,
                    seconds = elapsed,
                    avgUPS = if (elapsed > 0.0) outputSamples / elapsed else 0.0,
                ),
            ),
        )
        return HiftResult(outputSamples = outputSamples, output = hiftOutput)
    }

    private suspend fun runHifiGanInferenceFromBins(
        runner: MnnHifiGanRunner,
        inputDir: File,
        onEvent: ((InferenceEvent) -> Unit)? = null,
    ): HiftResult {
        fun emit(event: InferenceEvent) {
            onEvent?.invoke(event)
        }

        emit(
            InferenceEvent.StageProgress(
                stage = InferenceStage.hift,
                unitName = "HifiGan 推理中...",
                unitsDone = 1,
                secondsElapsed = 0.0,
                instUPS = 0.0,
                avgUPS = 0.0,
            ),
        )

        val input = readRawFloatArray(File(inputDir, FILE_HIFIGAN_INPUT))
        val shape = readRawIntShapeText(File(inputDir, FILE_HIFIGAN_INPUT_SHAPE))
        val t0 = nowSeconds()
        val hiftOutput = runner.forward(
            input = input,
            shape = shape,
        )
        require(hiftOutput.isNotEmpty()) { "HifiGan output is empty" }

        val outputSamples = hiftOutput.size
        val elapsed = nowSeconds() - t0
        emit(
            InferenceEvent.StageEnded(
                StageEndedInfo(
                    stage = InferenceStage.hift,
                    unitName = "HifiGan Inference completed",
                    units = outputSamples,
                    seconds = elapsed,
                    avgUPS = if (elapsed > 0.0) outputSamples / elapsed else 0.0,
                ),
            ),
        )
        return HiftResult(outputSamples = outputSamples, output = hiftOutput)
    }

    private fun runLLMInferenceFromPt(
        runner: MnnLlmRunner,
        resources: LocalResourceFiles,
        lmInputPtFile: File,
        onEvent: ((InferenceEvent) -> Unit)? = null,
    ): LlmOnlyOutput {
        fun emit(event: InferenceEvent) {
            onEvent?.invoke(event)
        }

        emit(InferenceEvent.StageBegan(InferenceStage.llm, "LLM token generation (llm_input.pt)"))
        runner.resetKvCache()
        val inputEmbeddings = readTorchPtInputsEmbeddings(lmInputPtFile)
        require(inputEmbeddings.isNotEmpty()) { "LM input embeddings are empty: ${lmInputPtFile.absolutePath}" }
        require(inputEmbeddings.size % LLM_HIDDEN_SIZE == 0) {
            "LM input embedding size ${inputEmbeddings.size} is not divisible by $LLM_HIDDEN_SIZE"
        }
        val promptSeqLen = inputEmbeddings.size / LLM_HIDDEN_SIZE
        require(promptSeqLen > 0) { "LM input seqLen must be > 0" }

        val llmBegin = nowSeconds()
        val params = ModelParameters()
        params.loadFromBinary(resources.speechEmbeddingWeight.absolutePath, "speech_embedding.weight")
        params.loadFromBinary(resources.llmDecoderWeight.absolutePath, "llm_decoder.weight")
        params.loadFromBinary(resources.llmDecoderBias.absolutePath, "llm_decoder.bias")

        val outTokens = ArrayList<Int>(LLM_ONLY_MAX_NEW_TOKENS.coerceAtMost(2048))
        val rng = SeededRNG(seed = 0L)
        var currentInput = inputEmbeddings
        var nPast = 0
        val reportEverySeconds = 1.0
        var lastReportT = llmBegin
        var lastReportCount = 0
        try {
            for (i in 0 until LLM_ONLY_MAX_NEW_TOKENS) {
                val llmRes = runner.decodeEmbeddings(currentInput, nPast)
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
                    ignoreEOS = i < LLM_ONLY_MIN_NEW_TOKENS,
                    rng = rng,
                )
                Log.i(
                    TAG,
                    "[LLM_ONLY] step=$i topId=$topId nPast=$nPast generated=${outTokens.size}",
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
        } finally {
            params.clear()
        }

        val elapsed = nowSeconds() - llmBegin
        emit(
            InferenceEvent.StageEnded(
                StageEndedInfo(
                    stage = InferenceStage.llm,
                    unitName = "LLM inference completed",
                    units = outTokens.size,
                    seconds = elapsed,
                    avgUPS = if (elapsed > 0.0) outTokens.size / elapsed else 0.0,
                ),
            ),
        )
        return LlmOnlyOutput(
            promptSeqLen = promptSeqLen,
            generatedTokens = outTokens.size,
        )
    }

    private suspend fun ensureHifiGanOnlyResourcesReady(): HifiGanOnlyResourceFiles =
        engineMutex.withLock {
            installBundledResourcesIfPresent()
            val modelsDir = ensureDirectory(File(appContext.filesDir, DIRECTORY_MODELS))
            val resourcesDir = ensureDirectory(File(appContext.filesDir, DIRECTORY_LOCAL_RESOURCES))
            HifiGanOnlyResourceFiles(
                hifiganModel = resolveRequiredFile(FILE_HIFIGAN_MODEL, listOf(File(modelsDir, FILE_HIFIGAN_MODEL))),
                hifiganInputsBinDir = resolveRequiredDirectory(
                    DIR_HIFIGAN_INPUTS_BIN,
                    listOf(File(resourcesDir, DIR_HIFIGAN_INPUTS_BIN)),
                ),
            )
        }

    private suspend fun ensureFlowOnlyResourcesReady(): FlowOnlyResourceFiles =
        engineMutex.withLock {
            installBundledResourcesIfPresent()
            val modelsDir = ensureDirectory(File(appContext.filesDir, DIRECTORY_MODELS))
            val resourcesDir = ensureDirectory(File(appContext.filesDir, DIRECTORY_LOCAL_RESOURCES))
            val optionalNoisePe = resolveOptionalFile(listOf(File(resourcesDir, FILE_NOISE_PE)))
            val flowInputsBinDir = resolveRequiredDirectory(
                DIR_FLOW_INPUTS_BIN,
                listOf(File(resourcesDir, DIR_FLOW_INPUTS_BIN)),
            )
            val flowLiteRtModel = resolveRequiredFile(
                FILE_FLOW_TFLITE_MODEL,
                listOf(File(modelsDir, FILE_FLOW_TFLITE_MODEL)),
            )

            if (FLOW_ONLY_BACKEND == FlowOnlyBackend.LITERT || FLOW_ONLY_BACKEND == FlowOnlyBackend.LITERT_CPP) {
                FlowOnlyResourceFiles(
                    flowModel = flowLiteRtModel,
                    flowEncoderModel = flowLiteRtModel,
                    flowDecoderModel = flowLiteRtModel,
                    flowQnnModel = flowLiteRtModel,
                    flowLiteRtModel = flowLiteRtModel,
                    noisePe = optionalNoisePe ?: flowLiteRtModel,
                    flowInputsBinDir = flowInputsBinDir,
                )
            } else if (FLOW_ONLY_BACKEND == FlowOnlyBackend.LLAMA_CPP) {
                val flowModel = resolveRequiredFile(
                    FILE_FLOW_GGUF_MODEL,
                    listOf(
                        File(modelsDir, FILE_FLOW_GGUF_MODEL),
                        File(modelsDir, FILE_LLM_MODEL),
                    ),
                )
                val noisePe = resolveRequiredFile(FILE_NOISE_PE, listOf(File(resourcesDir, FILE_NOISE_PE)))
                FlowOnlyResourceFiles(
                    flowModel = flowModel,
                    flowEncoderModel = flowModel,
                    flowDecoderModel = flowModel,
                    flowQnnModel = flowModel,
                    flowLiteRtModel = flowLiteRtModel,
                    noisePe = noisePe,
                    flowInputsBinDir = flowInputsBinDir,
                )
            } else {
                val mnnModelsDir = ensureDirectory(File(modelsDir, DIR_MNN_MODELS))
                val flowModel = resolveRequiredFile(
                    FILE_FLOW_GGUF_MODEL,
                    listOf(
                        File(modelsDir, FILE_FLOW_GGUF_MODEL),
                        File(modelsDir, FILE_LLM_MODEL),
                    ),
                )
                FlowOnlyResourceFiles(
                    flowModel = flowModel,
                    flowEncoderModel = resolveRequiredFile(
                        "$DIR_MNN_MODELS/$FILE_FLOW_ENCODER_MODEL",
                        listOf(File(mnnModelsDir, FILE_FLOW_ENCODER_MODEL)),
                    ),
                    flowDecoderModel = resolveRequiredFile(
                        "$DIR_MNN_MODELS/$FILE_FLOW_DECODER_MODEL_TEST_OP",
                        listOf(
                            File(mnnModelsDir, FILE_FLOW_DECODER_MODEL_TEST_OP),
                            File(mnnModelsDir, FILE_FLOW_DECODER_MODEL_GPU_SIMPLIFIED),
                            File(mnnModelsDir, FILE_FLOW_DECODER_MODEL),
                        ),
                    ),
                    flowQnnModel = resolveRequiredFile(
                        FILE_FLOW_QNN_ONNX_MODEL,
                        listOf(File(modelsDir, FILE_FLOW_QNN_ONNX_MODEL)),
                    ),
                    flowLiteRtModel = flowLiteRtModel,
                    noisePe = optionalNoisePe ?: flowLiteRtModel,
                    flowInputsBinDir = flowInputsBinDir,
                )
            }
        }

    private fun inferHifiGanInputShape(flatSize: Int): IntArray {
        require(flatSize > 0) { "HifiGan input size must be > 0" }
        require(flatSize % HIFIGAN_MEL_BINS == 0) {
            "Invalid HifiGan input size: $flatSize, must be divisible by $HIFIGAN_MEL_BINS"
        }
        val frames = flatSize / HIFIGAN_MEL_BINS
        return intArrayOf(1, HIFIGAN_MEL_BINS, frames)
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

    private fun readRawInt32Array(file: File): IntArray {
        val bytes = file.readBytes()
        require(bytes.size % 4 == 0) { "Invalid int32 bin size: ${file.absolutePath}, bytes=${bytes.size}" }
        val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val out = IntArray(bytes.size / 4)
        for (i in out.indices) out[i] = bb.int
        return out
    }

    private fun readRawInt64Array(file: File): LongArray {
        val bytes = file.readBytes()
        require(bytes.size % 8 == 0) { "Invalid int64 bin size: ${file.absolutePath}, bytes=${bytes.size}" }
        val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val out = LongArray(bytes.size / 8)
        for (i in out.indices) out[i] = bb.long
        return out
    }

    private fun readRawFloatArray(file: File): FloatArray {
        val bytes = file.readBytes()
        require(bytes.size % 4 == 0) { "Invalid float32 bin size: ${file.absolutePath}, bytes=${bytes.size}" }
        val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val out = FloatArray(bytes.size / 4)
        for (i in out.indices) out[i] = bb.float
        return out
    }

    private fun readRawInt32Scalar(file: File): Int {
        val bytes = file.readBytes()
        require(bytes.size == 4) { "Invalid int32 scalar bin size: ${file.absolutePath}, bytes=${bytes.size}" }
        return ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).int
    }

    private fun readRawIntShapeText(file: File): IntArray {
        val raw = file.readText(Charsets.UTF_8).trim()
        require(raw.isNotEmpty()) { "Empty shape text: ${file.absolutePath}" }
        val tokens = raw.split(Regex("[,xX\\s]+")).filter { it.isNotBlank() }
        require(tokens.isNotEmpty()) { "Invalid shape text: ${file.absolutePath}, text=$raw" }
        val shape = IntArray(tokens.size)
        for (i in tokens.indices) {
            val dim = tokens[i].toIntOrNull()
                ?: error("Invalid shape dim '${tokens[i]}' in ${file.absolutePath}")
            require(dim > 0) { "Shape dim must be > 0 in ${file.absolutePath}, got $dim" }
            shape[i] = dim
        }
        return shape
    }

    private fun readRawBoolByte(file: File): Boolean {
        val bytes = file.readBytes()
        require(bytes.size == 1) { "Invalid bool bin size: ${file.absolutePath}, bytes=${bytes.size}" }
        return bytes[0].toInt() != 0
    }

    private fun readTorchPtInputsEmbeddings(ptFile: File): FloatArray {
        require(ptFile.exists() && ptFile.isFile) { "llm_input.pt not found: ${ptFile.absolutePath}" }
        ZipFile(ptFile).use { zip ->
            val allEntries = ArrayList<java.util.zip.ZipEntry>()
            val enumeration = zip.entries()
            while (enumeration.hasMoreElements()) {
                allEntries.add(enumeration.nextElement())
            }

            val candidates = allEntries
                .filter { !it.isDirectory && it.name.contains("/data/") && it.size > 0L }
                .sortedByDescending { it.size }
            require(candidates.isNotEmpty()) {
                "No tensor storage entries found in ${ptFile.absolutePath}"
            }
            val preferred = candidates.filter { it.name.endsWith("/data/0") }
            val selectionPool = if (preferred.isNotEmpty()) preferred else candidates

            val selected = selectionPool.firstOrNull { entry ->
                val bytes = entry.size
                bytes % 4L == 0L &&
                    ((bytes / 4L) % LLM_HIDDEN_SIZE.toLong() == 0L)
            } ?: error(
                "No compatible float storage found in ${ptFile.absolutePath}, candidates=${selectionPool.map { it.name }}",
            )

            val bytes = zip.getInputStream(selected).use { it.readBytes() }
            require(bytes.size % 4 == 0) { "Invalid float data bytes in ${selected.name}: ${bytes.size}" }
            val out = FloatArray(bytes.size / 4)
            val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            for (i in out.indices) {
                out[i] = bb.float
            }
            return out
        }
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

    private fun buildDecoderHeadPreviewNote(output: FloatArray): String {
        if (output.isEmpty()) {
            return "[DecoderOut] [0,0,:8]=[] (empty output)"
        }
        val count = min(8, output.size)
        val values = (0 until count).joinToString(
            separator = ", ",
            prefix = "[",
            postfix = "]",
        ) { i -> String.format(Locale.US, "%.6f", output[i]) }
        return "[DecoderOut] [0,0,:8]=$values"
    }

    private suspend fun ensureEngineReady(modelFile: File, flowModelFile: File): InferenceEngine =
        engineMutex.withLock {
            engine?.let { return it }

            installBundledResourcesIfPresent()
            Log.i(TAG, "[LLMPrepare] get InferenceEngine instance")
            val loaded = AiChat.getInferenceEngine(appContext)
            val state = loaded.state.value
            Log.i(TAG, "[LLMPrepare] current engine state=${state.javaClass.simpleName}")
            if (state.isModelLoaded || state is InferenceEngine.State.Error) {
                Log.i(TAG, "[LLMPrepare] cleanUp old engine state")
                loaded.cleanUp()
                Log.i(TAG, "[LLMPrepare] cleanUp done")
            }
            val t0 = nowSeconds()
            Log.i(TAG, "[LLMPrepare] loadModel begin: ${modelFile.absolutePath} (bytes=${modelFile.length()})")
            loaded.loadModel(modelFile.absolutePath)
            Log.i(TAG, "[LLMPrepare] loadModel done in ${"%.3f".format(Locale.US, nowSeconds() - t0)} s")

            val t1 = nowSeconds()
            Log.i(TAG, "[LLMPrepare] loadFlowModel begin: ${flowModelFile.absolutePath} (bytes=${flowModelFile.length()})")
            loaded.loadFlowModel(flowModelFile.absolutePath)
            Log.i(TAG, "[LLMPrepare] loadFlowModel done in ${"%.3f".format(Locale.US, nowSeconds() - t1)} s")
            engine = loaded
            loaded
        }

    private suspend fun ensureFlowOnlyEngineReady(flowModelFile: File): InferenceEngine =
        engineMutex.withLock {
            engine?.let { return it }

            installBundledResourcesIfPresent()
            Log.i(TAG, "[FlowOnly] get InferenceEngine instance")
            val loaded = AiChat.getInferenceEngine(appContext)
            val state = loaded.state.value
            if (state.isModelLoaded || state is InferenceEngine.State.Error) {
                loaded.cleanUp()
            }
            val t0 = nowSeconds()
            Log.i(TAG, "[FlowOnly] loadFlowModel begin: ${flowModelFile.absolutePath} (bytes=${flowModelFile.length()})")
            loaded.loadFlowModel(flowModelFile.absolutePath)
            Log.i(TAG, "[FlowOnly] loadFlowModel done in ${"%.3f".format(Locale.US, nowSeconds() - t0)} s")
            engine = loaded
            loaded
        }

    private suspend fun ensureLiteRtFlowRunnerReady(flowLiteRtModelFile: File): LiteRtFlowRunner =
        engineMutex.withLock {
            liteRtFlowRunner?.let { return it }

            installBundledResourcesIfPresent()
            val t0 = nowSeconds()
            Log.i(
                TAG,
                "[FlowOnly][LiteRT] load begin: ${flowLiteRtModelFile.absolutePath} (bytes=${flowLiteRtModelFile.length()})",
            )
            val loaded = LiteRtFlowRunner.load(modelFile = flowLiteRtModelFile)
            Log.i(
                TAG,
                "[FlowOnly][LiteRT] load done in ${"%.3f".format(Locale.US, nowSeconds() - t0)} s, runtime=${loaded.runtimeMode}",
            )
            liteRtFlowRunner = loaded
            loaded
        }

    private suspend fun ensureLiteRtNativeFlowRunnerReady(flowLiteRtModelFile: File): LiteRtNativeFlowRunner =
        engineMutex.withLock {
            liteRtNativeFlowRunner?.let { return it }

            installBundledResourcesIfPresent()
            val t0 = nowSeconds()
            Log.i(
                TAG,
                "[FlowOnly][LiteRT][C++] load begin: ${flowLiteRtModelFile.absolutePath} (bytes=${flowLiteRtModelFile.length()})",
            )
            val loaded = LiteRtNativeFlowRunner.load(modelFile = flowLiteRtModelFile)
            Log.i(
                TAG,
                "[FlowOnly][LiteRT][C++] load done in ${"%.3f".format(Locale.US, nowSeconds() - t0)} s, runtime=${loaded.runtimeMode}",
            )
            liteRtNativeFlowRunner = loaded
            loaded
        }

    private suspend fun ensureMnnFlowEncoderRunnerReady(flowEncoderModelFile: File): MnnFlowRunner =
        engineMutex.withLock {
            mnnFlowEncoderRunner?.let { return it }

            installBundledResourcesIfPresent()
            val t0 = nowSeconds()
            Log.i(
                TAG,
                "[FlowOnly][MNN] encoder load begin: ${flowEncoderModelFile.absolutePath} (bytes=${flowEncoderModelFile.length()})",
            )
            val loaded = MnnFlowRunner.load(
                modelFile = flowEncoderModelFile,
                enableOpProfile = FLOW_MNN_OP_PROFILE_ENABLED,
            )
            Log.i(
                TAG,
                "[FlowOnly][MNN] encoder load done in ${"%.3f".format(Locale.US, nowSeconds() - t0)} s",
            )
            mnnFlowEncoderRunner = loaded
            loaded
        }

    private suspend fun ensureMnnFlowDecoderRunnerReady(flowDecoderModelFile: File): MnnFlowDecoderRunner =
        engineMutex.withLock {
            mnnFlowDecoderRunner?.let { return it }

            installBundledResourcesIfPresent()
            val t0 = nowSeconds()
            Log.i(
                TAG,
                "[FlowOnly][MNN] decoder load begin: ${flowDecoderModelFile.absolutePath} (bytes=${flowDecoderModelFile.length()})",
            )
            val loaded = MnnFlowDecoderRunner.load(
                modelFile = flowDecoderModelFile,
                enableOpProfile = FLOW_MNN_OP_PROFILE_ENABLED,
            )
            Log.i(
                TAG,
                "[FlowOnly][MNN] decoder load done in ${"%.3f".format(Locale.US, nowSeconds() - t0)} s",
            )
            mnnFlowDecoderRunner = loaded
            loaded
        }

    private suspend fun ensureOnnxQnnFlowRunnerReady(flowQnnModelFile: File): OnnxQnnFlowRunner =
        engineMutex.withLock {
            onnxQnnFlowRunner?.let { return it }

            installBundledResourcesIfPresent()
            val t0 = nowSeconds()
            Log.i(
                TAG,
                "[FlowOnly][ONNX-QNN] session load begin: ${flowQnnModelFile.absolutePath} (bytes=${flowQnnModelFile.length()})",
            )
            val loaded = OnnxQnnFlowRunner.load(
                modelFile = flowQnnModelFile,
                backendPath = DEFAULT_QNN_GPU_BACKEND_PATH,
                disableCpuFallback = false,
            )
            Log.i(
                TAG,
                "[FlowOnly][ONNX-QNN] session load done in ${"%.3f".format(Locale.US, nowSeconds() - t0)} s, " +
                    "disableCpuFallback=${loaded.cpuFallbackDisabled}",
            )
            onnxQnnFlowRunner = loaded
            loaded
        }

    private suspend fun ensureMnnHifiGanRunnerReady(hifiganModelFile: File): MnnHifiGanRunner =
        engineMutex.withLock {
            mnnHifiGanRunner?.let { return it }

            installBundledResourcesIfPresent()
            val t0 = nowSeconds()
            Log.i(
                TAG,
                "[HifiGan][MNN] module load begin: ${hifiganModelFile.absolutePath} (bytes=${hifiganModelFile.length()})",
            )
            val loaded = MnnHifiGanRunner.load(modelFile = hifiganModelFile)
            Log.i(
                TAG,
                "[HifiGan][MNN] module load done in ${"%.3f".format(Locale.US, nowSeconds() - t0)} s",
            )
            mnnHifiGanRunner = loaded
            loaded
        }

    private suspend fun ensureMnnLlmRunnerReady(resources: LocalResourceFiles): MnnLlmRunner =
        engineMutex.withLock {
            mnnLlmRunner?.let { return it }

            val configFile = resources.mnnLlmConfig
                ?: error(
                    "MNN LLM config not found. Expected one of:\n" +
                        "- ${File(appContext.filesDir, "$DIRECTORY_MODELS/$DIR_MNN_MODELS/$FILE_MNN_LLM_CONFIG").absolutePath}\n" +
                        "- ${File(appContext.filesDir, "$DIRECTORY_MODELS/$FILE_MNN_LLM_CONFIG").absolutePath}",
                )
            val t0 = nowSeconds()
            Log.i(TAG, "[MNN-LLM] load begin: ${configFile.absolutePath} (bytes=${configFile.length()})")
            val loaded = MnnLlmRunner.load(configFile = configFile)
            Log.i(TAG, "[MNN-LLM] load done in ${"%.3f".format(Locale.US, nowSeconds() - t0)} s")
            mnnLlmRunner = loaded
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
        ensureCriticalResourceFilesSynced(
            assetDir = ASSET_DIR_MODELS,
            targetDir = targetModelsDir,
            requiredFileCandidates = listOf(
                listOf(FILE_FLOW_GGUF_MODEL, FILE_LLM_MODEL),
                listOf("$DIR_MNN_MODELS/$FILE_MNN_LLM_CONFIG", FILE_MNN_LLM_CONFIG),
                listOf("$DIR_MNN_MODELS/$FILE_FLOW_MODEL"),
                listOf("$DIR_MNN_MODELS/$FILE_FLOW_ENCODER_MODEL"),
                listOf(
                    "$DIR_MNN_MODELS/$FILE_FLOW_DECODER_MODEL_TEST_OP",
                    "$DIR_MNN_MODELS/$FILE_FLOW_DECODER_MODEL",
                ),
                listOf(
                    "$DIR_MNN_MODELS/$FILE_FLOW_MODEL_WEIGHTS",
                    "$DIR_MNN_MODELS/$FILE_FLOW_MODEL_WEIGHT",
                ),
                listOf(FILE_FLOW_QNN_ONNX_MODEL),
                listOf(FILE_FLOW_TFLITE_MODEL),
            ),
        )

        val targetResourcesDir = File(appContext.filesDir, DIRECTORY_LOCAL_RESOURCES).also {
            if (it.exists() && !it.isDirectory) it.delete()
            if (!it.exists()) it.mkdirs()
        }
        copyAssetDirectoryIfPresent(assetDir = ASSET_DIR_LOCAL_RESOURCES, targetDir = targetResourcesDir)
        ensureCriticalResourceFilesSynced(
            assetDir = ASSET_DIR_LOCAL_RESOURCES,
            targetDir = targetResourcesDir,
            requiredFileCandidates = listOf(
                listOf(FILE_LLM_INPUT_PT, FILE_LLM_INPUT_PT_LEGACY, FILE_LLM_INPUT_PT_UPPER_LEGACY),
                listOf("$DIR_FLOW_INPUTS_BIN/$FILE_FLOW_DECODER_INPUT_PT"),
            ),
        )
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
        val expectedToken = currentAssetSyncToken(assetDir)
        if (markerFile.exists()) {
            val recorded = runCatching { markerFile.readText() }.getOrNull()?.trim().orEmpty()
            if (recorded == expectedToken) {
                return
            }
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
        markerFile.writeText(expectedToken)
    }

    private fun currentAssetSyncToken(assetDir: String): String {
        val pkgInfo = appContext.packageManager.getPackageInfo(appContext.packageName, 0)
        return "dir=$assetDir;updatedAt=${pkgInfo.lastUpdateTime};version=${pkgInfo.longVersionCode}"
    }

    private fun ensureCriticalResourceFilesSynced(
        assetDir: String,
        targetDir: File,
        requiredFileCandidates: List<List<String>>,
    ) {
        val hasMissing = requiredFileCandidates.any { candidates ->
            candidates.none { name -> File(targetDir, name).isFile }
        }
        if (!hasMissing) return

        clearSyncMarkersRecursively(targetDir)
        copyAssetDirectoryIfPresent(assetDir = assetDir, targetDir = targetDir)
    }

    private fun clearSyncMarkersRecursively(dir: File) {
        if (!dir.exists() || !dir.isDirectory) return
        val marker = File(dir, ASSET_SYNC_MARKER)
        if (marker.exists()) {
            marker.delete()
        }
        dir.listFiles()?.forEach { child ->
            if (child.isDirectory) {
                clearSyncMarkersRecursively(child)
            }
        }
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
            val mnnModelsDir = ensureDirectory(File(modelsDir, DIR_MNN_MODELS))

            val resolved = LocalResourceFiles(
                llmModel = resolveOptionalFile(listOf(File(modelsDir, FILE_LLM_MODEL))) ?: resolveModelFile(),
                mnnLlmConfig = resolveOptionalFile(
                    listOf(
                        File(modelsDir, "$DIR_MNN_MODELS/$FILE_MNN_LLM_CONFIG"),
                        File(modelsDir, FILE_MNN_LLM_CONFIG),
                    ),
                ),
                flowModel = resolveRequiredFile(
                    "$DIR_MNN_MODELS/$FILE_FLOW_MODEL",
                    listOf(File(mnnModelsDir, FILE_FLOW_MODEL)),
                ),
                flowModelWeight = resolveRequiredFile(
                    "$DIR_MNN_MODELS/$FILE_FLOW_MODEL_WEIGHTS",
                    listOf(
                        File(mnnModelsDir, FILE_FLOW_MODEL_WEIGHTS),
                        File(mnnModelsDir, FILE_FLOW_MODEL_WEIGHT),
                    ),
                ),
                hifiganModel = resolveRequiredFile(FILE_HIFIGAN_MODEL, listOf(File(modelsDir, FILE_HIFIGAN_MODEL))),
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
                lmInputPt = resolveRequiredFile(
                    FILE_LLM_INPUT_PT,
                    listOf(
                        File(resourcesDir, FILE_LLM_INPUT_PT),
                        File(resourcesDir, FILE_LLM_INPUT_PT_LEGACY),
                        File(resourcesDir, FILE_LLM_INPUT_PT_UPPER_LEGACY),
                    ),
                ),
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
                flowInputsBinDir = resolveRequiredDirectory(DIR_FLOW_INPUTS_BIN, listOf(File(resourcesDir, DIR_FLOW_INPUTS_BIN))),
                hifiganInputsBinDir = resolveRequiredDirectory(DIR_HIFIGAN_INPUTS_BIN, listOf(File(resourcesDir, DIR_HIFIGAN_INPUTS_BIN))),
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

    private enum class FlowOnlyBackend {
        LLAMA_CPP,
        MNN_ENCODER_DECODER,
        ONNX_QNN_GPU,
        LITERT_CPP,
        LITERT,
    }

    companion object {
        private const val TAG = "LlamaInferenceBridge"
        private const val USE_MNN_LLM_BACKEND = false
        private const val LLM_ONLY_TEST_MODE = false
        private const val FLOW_ONLY_TEST_MODE = true
        private const val HIFIGAN_ONLY_TEST_MODE = false
        private val FLOW_ONLY_BACKEND = FlowOnlyBackend.LITERT
        private const val FLOW_MNN_OP_PROFILE_ENABLED = true
        private const val ASSET_SYNC_MARKER = ".asset_sync_ok"
        private const val DIRECTORY_MODELS = "models"
        private const val DIRECTORY_LOCAL_RESOURCES = "local_llm_resources"
        private const val ASSET_DIR_MODELS = "models"
        private const val ASSET_DIR_LOCAL_RESOURCES = "local_llm_resources"

        private const val FILE_LLM_MODEL = "flow_fp32.gguf"
        private const val FILE_FLOW_GGUF_MODEL = "flow_fp32.gguf"
        private const val DIR_MNN_MODELS = "mnnModels"
        private const val FILE_MNN_LLM_CONFIG = "config.json"
        private const val FILE_FLOW_MODEL = "flow.mnn"
        private const val FILE_FLOW_ENCODER_MODEL = "flow_encoder.mnn"
        private const val FILE_FLOW_DECODER_MODEL = "flow_decoder.mnn"
        private const val FILE_FLOW_DECODER_MODEL_TEST_OP = "flow_decoder_test_op.mnn"
        private const val FILE_FLOW_DECODER_MODEL_GPU_SIMPLIFIED = "flow_decoder_gpu_simplified.mnn"
        private const val FILE_FLOW_QNN_ONNX_MODEL = "flow_qnn_ort_opt.onnx"
        private const val FILE_FLOW_TFLITE_MODEL = "flow.tflite"
        private const val FILE_FLOW_MODEL_WEIGHTS = "flow.mnn.weights"
        private const val FILE_FLOW_MODEL_WEIGHT = "flow.mnn.weight"
        private const val FILE_HIFIGAN_MODEL = "hifigan.mnn"

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
        private const val FILE_LLM_INPUT_PT = "llm_input.pt"
        private const val FILE_LLM_INPUT_PT_LEGACY = "lm_input.pt"
        private const val FILE_LLM_INPUT_PT_UPPER_LEGACY = "LLM_input.pt"
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
        private const val DIR_FLOW_INPUTS_BIN = "flow_inputs_bin"
        private const val DIR_HIFIGAN_INPUTS_BIN = "hifigan_inputs_bin"

        private const val FILE_HIFIGAN_INPUT = "0_input.bin"
        private const val FILE_HIFIGAN_INPUT_SHAPE = "0_input.shape.txt"

        private const val FILE_FLOW_INPUT_TOKEN = "0_token.bin"
        private const val FILE_FLOW_INPUT_TOKEN_LEN = "1_token_len.bin"
        private const val FILE_FLOW_INPUT_PROMPT_TOKEN = "2_prompt_token.bin"
        private const val FILE_FLOW_INPUT_PROMPT_TOKEN_LEN = "3_prompt_token_len.bin"
        private const val FILE_FLOW_INPUT_PROMPT_FEAT = "4_prompt_feat.bin"
        private const val FILE_FLOW_INPUT_PROMPT_FEAT_LEN = "5_prompt_feat_len.bin"
        private const val FILE_FLOW_INPUT_EMBEDDING = "6_embedding.bin"
        private const val FILE_FLOW_INPUT_STREAMING = "7_streaming.bin"
        private const val FILE_FLOW_INPUT_FINALIZE = "8_finalize.bin"
        private const val FILE_FLOW_DECODER_INPUT_PT = "flow_decoder_inputs.pt"

        private const val DEFAULT_SYSTEM_PROMPT =
            "You are a concise assistant. Return plain text suitable for speech output."
        private const val LLM_ONLY_MIN_NEW_TOKENS = 0
        private const val LLM_ONLY_MAX_NEW_TOKENS = 512
        private const val LLM_HIDDEN_SIZE = 896
        private const val FLOW_DECODER_MEL_BINS = 80
        private const val HIFT_N_FFT = 16
        private const val HIFT_HOP_LENGTH = 4
        private const val HIFT_N_FREQ = HIFT_N_FFT / 2 + 1
        private const val HIFT_FRAME_FEATURES = HIFT_N_FREQ * 2
        private const val HIFIGAN_MEL_BINS = 80
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

private data class LlmOnlyOutput(
    val promptSeqLen: Int,
    val generatedTokens: Int,
)

private data class LocalResourceFiles(
    val llmModel: File,
    val mnnLlmConfig: File?,
    val flowModel: File,
    val flowModelWeight: File,
    val hifiganModel: File,
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
    val lmInputPt: File,
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
    val flowInputsBinDir: File,
    val hifiganInputsBinDir: File,
)

private data class FlowOnlyResourceFiles(
    val flowModel: File,
    val flowEncoderModel: File,
    val flowDecoderModel: File,
    val flowQnnModel: File,
    val flowLiteRtModel: File,
    val noisePe: File,
    val flowInputsBinDir: File,
)

private enum class FlowDecoderInputMode {
    Legacy5,
    Prepared6,
}

private data class FlowDecoderInputBundle(
    val mode: FlowDecoderInputMode,
    val mu: FloatArray = FloatArray(0),
    val mask: FloatArray = FloatArray(0),
    val z: FloatArray = FloatArray(0),
    val spks: FloatArray = FloatArray(0),
    val cond: FloatArray = FloatArray(0),
    val xIn: FloatArray = FloatArray(0),
    val maskIn: FloatArray = FloatArray(0),
    val muIn: FloatArray = FloatArray(0),
    val tIn: FloatArray = FloatArray(0),
    val spksIn: FloatArray = FloatArray(0),
    val condIn: FloatArray = FloatArray(0),
) {
    fun describeForLog(): String =
        when (mode) {
            FlowDecoderInputMode.Legacy5 ->
                "[EncDecOnly][MNN] decoder pt loaded, mode=legacy5, mu=${mu.size}, mask=${mask.size}, " +
                    "z=${z.size}, spks=${spks.size}, cond=${cond.size}"
            FlowDecoderInputMode.Prepared6 ->
                "[EncDecOnly][MNN] decoder pt loaded, mode=prepared6, x_in=${xIn.size}, mask_in=${maskIn.size}, " +
                    "mu_in=${muIn.size}, t_in=${tIn.size}, spks_in=${spksIn.size}, cond_in=${condIn.size}"
        }
}

private data class FlowOnlyInputBundle(
    val flowToken: LongArray,
    val tokenLen: Int,
    val promptToken: LongArray,
    val promptTokenLen: Int,
    val promptFeat: FloatArray,
    val promptFeatLen: Int,
    val embedding: FloatArray,
    val streaming: Boolean,
    val finalize: Boolean,
)

private data class HifiGanOnlyResourceFiles(
    val hifiganModel: File,
    val hifiganInputsBinDir: File,
)
