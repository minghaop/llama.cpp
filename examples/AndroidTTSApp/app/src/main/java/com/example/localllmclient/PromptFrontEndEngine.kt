package com.example.llama

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.IntBuffer
import kotlin.math.ln
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.min

internal data class PromptSpeechArtifactsAndroid(
    val speechTokens: IntArray,
    val speechTokenLen: Int,
    val speechFeat: FloatArray,
    val speechFeatLen: Int,
    val speechEmbedding: FloatArray,
)

internal class PromptFrontEndEngine(
    speechTokenizerModel: File,
    campPlusModel: File,
    mel128Bin: File,
    hannWindow400Bin: File,
    hannWindow1920Bin: File,
    librosaMelFnBin: File,
    fftBasisRealBin: File,
    fftBasisImagBin: File,
    poveyWindow400Bin: File,
    melBanks80x257Bin: File,
) : AutoCloseable {
    private val mel128 = loadRawFloatArray(mel128Bin, N_MELS_TOKEN * FFT_BINS_TOKEN)
    private val hannWindow400 = loadRawFloatArray(hannWindow400Bin, N_FFT_TOKEN)
    private val hannWindow1920 = loadRawFloatArray(hannWindow1920Bin, FLOW_N_FFT)
    private val librosaMelFn = loadRawFloatArray(librosaMelFnBin, FLOW_NUM_MELS * FLOW_FFT_BINS)
    private val fftBasisReal = loadRawFloatArray(fftBasisRealBin, FFT_BINS_TOKEN * N_FFT_TOKEN)
    private val fftBasisImag = loadRawFloatArray(fftBasisImagBin, FFT_BINS_TOKEN * N_FFT_TOKEN)
    private val poveyWindow400 = loadRawFloatArray(poveyWindow400Bin, KALDI_WINDOW_SIZE)
    private val melBanks80x257 = loadRawFloatArray(melBanks80x257Bin, KALDI_NUM_MELS * KALDI_FFT_BINS)

    private val ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val speechTokenizerOptions = OrtSession.SessionOptions()
    private val campPlusOptions = OrtSession.SessionOptions()
    private val speechTokenizerSession: OrtSession
    private val campPlusSession: OrtSession

    init {
        speechTokenizerOptions.setIntraOpNumThreads(1)
        speechTokenizerOptions.setInterOpNumThreads(1)
        campPlusOptions.setIntraOpNumThreads(1)
        campPlusOptions.setInterOpNumThreads(1)

        speechTokenizerSession = ortEnv.createSession(speechTokenizerModel.absolutePath, speechTokenizerOptions)
        campPlusSession = ortEnv.createSession(campPlusModel.absolutePath, campPlusOptions)
    }

    fun build(promptAudio: FloatArray, promptSampleRate: Int): PromptSpeechArtifactsAndroid {
        val promptAudio16k = sanitizeAudio(resampleLinear(promptAudio, promptSampleRate, TARGET_SAMPLE_RATE))
        val promptAudio24k = sanitizeAudio(resampleLinear(promptAudio16k, TARGET_SAMPLE_RATE, TARGET_RESAMPLE_RATE))

        val (speechFeatFlat, speechFeatFrames) = extractFlowSpeechFeat(promptAudio24k)
        val speechTokens = extractSpeechToken(promptAudio16k)
        val speechEmbedding = extractCampPlusEmbedding(promptAudio16k)

        val alignedTokenLen = min(speechTokens.size, speechFeatFrames / 2).coerceAtLeast(0)
        val alignedTokens = if (alignedTokenLen == speechTokens.size) {
            speechTokens
        } else {
            speechTokens.copyOf(alignedTokenLen)
        }
        val alignedFeatLen = alignedTokenLen * 2
        val featRequired = alignedFeatLen * FLOW_NUM_MELS
        val alignedFeat = if (featRequired > 0 && speechFeatFlat.size >= featRequired) {
            speechFeatFlat.copyOfRange(0, featRequired)
        } else {
            speechFeatFlat
        }

        return PromptSpeechArtifactsAndroid(
            speechTokens = alignedTokens,
            speechTokenLen = alignedTokenLen,
            speechFeat = alignedFeat,
            speechFeatLen = alignedFeatLen,
            speechEmbedding = speechEmbedding,
        )
    }

    override fun close() {
        runCatching { speechTokenizerSession.close() }
        runCatching { campPlusSession.close() }
        runCatching { speechTokenizerOptions.close() }
        runCatching { campPlusOptions.close() }
    }

    private fun extractSpeechToken(promptAudio16k: FloatArray): IntArray {
        val (feat, frames) = whisperLogMelSpectrogram(promptAudio16k)
        if (frames <= 0 || feat.isEmpty()) return IntArray(0)

        val inputNames = speechTokenizerSession.inputNames.toList()
        require(inputNames.size >= 2) { "speech_tokenizer_v2.onnx expects >= 2 inputs" }
        val featInputName = inputNames.firstOrNull {
            it.contains("feat", ignoreCase = true) && !it.contains("len", ignoreCase = true)
        } ?: inputNames.first()
        val featLenInputName = inputNames.firstOrNull { it.contains("len", ignoreCase = true) }
            ?: inputNames.firstOrNull { it != featInputName }
            ?: inputNames.first()
        val outputName = speechTokenizerSession.outputNames.firstOrNull()
            ?: error("speech_tokenizer_v2.onnx has no output")

        OnnxTensor.createTensor(
            ortEnv,
            FloatBuffer.wrap(feat),
            longArrayOf(1, N_MELS_TOKEN.toLong(), frames.toLong()),
        ).use { featTensor ->
            OnnxTensor.createTensor(
                ortEnv,
                IntBuffer.wrap(intArrayOf(frames)),
                longArrayOf(1),
            ).use { featLengthTensor ->
                val inputs = HashMap<String, OnnxTensor>(2)
                inputs[featInputName] = featTensor
                inputs[featLenInputName] = featLengthTensor

                speechTokenizerSession.run(inputs, setOf(outputName)).use { outputs ->
                    val outputValue = outputs[0].value
                    return flattenToIntArray(outputValue)
                }
            }
        }
    }

    private fun extractCampPlusEmbedding(promptAudio16k: FloatArray): FloatArray {
        val (fbankFeat, frameCount) = extractKaldiFbank(promptAudio16k)
        if (frameCount <= 0 || fbankFeat.isEmpty()) return FloatArray(0)

        val inputName = campPlusSession.inputNames.firstOrNull()
            ?: error("campplus.onnx has no input")
        val outputName = campPlusSession.outputNames.firstOrNull()
            ?: error("campplus.onnx has no output")

        OnnxTensor.createTensor(
            ortEnv,
            FloatBuffer.wrap(fbankFeat),
            longArrayOf(1, frameCount.toLong(), KALDI_NUM_MELS.toLong()),
        ).use { inputTensor ->
            val inputs = hashMapOf(inputName to inputTensor)
            campPlusSession.run(inputs, setOf(outputName)).use { outputs ->
                return flattenToFloatArray(outputs[0].value)
            }
        }
    }

    private fun whisperLogMelSpectrogram(speech: FloatArray): Pair<FloatArray, Int> {
        if (speech.size < (N_FFT_TOKEN + 2)) return FloatArray(0) to 0
        val frames = speech.size / HOP_LENGTH_TOKEN
        if (frames <= 0) return FloatArray(0) to 0

        val padded = reflectPad(speech, N_FFT_TOKEN / 2, N_FFT_TOKEN / 2)
        val windowed = FloatArray(N_FFT_TOKEN)
        val magnitudeSq = FloatArray(FFT_BINS_TOKEN * frames)

        var t = 0
        while (t < frames) {
            val start = t * HOP_LENGTH_TOKEN
            var i = 0
            while (i < N_FFT_TOKEN) {
                windowed[i] = padded[start + i] * hannWindow400[i]
                i += 1
            }

            var k = 0
            while (k < FFT_BINS_TOKEN) {
                val basisOffset = k * N_FFT_TOKEN
                var real = 0f
                var imag = 0f
                i = 0
                while (i < N_FFT_TOKEN) {
                    val sample = windowed[i]
                    real += sample * fftBasisReal[basisOffset + i]
                    imag += sample * fftBasisImag[basisOffset + i]
                    i += 1
                }
                magnitudeSq[k * frames + t] = real * real + imag * imag
                k += 1
            }
            t += 1
        }

        val melSpec = FloatArray(N_MELS_TOKEN * frames)
        var m = 0
        while (m < N_MELS_TOKEN) {
            val melOffset = m * FFT_BINS_TOKEN
            t = 0
            while (t < frames) {
                var sum = 0f
                var k = 0
                while (k < FFT_BINS_TOKEN) {
                    sum += mel128[melOffset + k] * magnitudeSq[k * frames + t]
                    k += 1
                }
                melSpec[m * frames + t] = sum
                t += 1
            }
            m += 1
        }

        var maxVal = Float.NEGATIVE_INFINITY
        for (v in melSpec) {
            if (v > maxVal) maxVal = v
        }
        val clampMin = 1e-10f
        val maxLogVal = log10(max(maxVal, clampMin))
        val minLogVal = maxLogVal - 8f
        var idx = 0
        while (idx < melSpec.size) {
            val clamped = max(melSpec[idx], clampMin)
            val logged = log10(clamped)
            melSpec[idx] = (max(logged, minLogVal) + 4f) / 4f
            idx += 1
        }

        return melSpec to frames
    }

    private fun extractFlowSpeechFeat(prompt24k: FloatArray): Pair<FloatArray, Int> {
        if (prompt24k.size <= FLOW_PAD + 2) return FloatArray(0) to 0

        val padded = reflectPadTorch1D(prompt24k, FLOW_PAD)
        if (padded.size < FLOW_N_FFT) return FloatArray(0) to 0

        val frames = 1 + (padded.size - FLOW_N_FFT) / FLOW_HOP
        if (frames <= 0) return FloatArray(0) to 0

        val mag = FloatArray(FLOW_FFT_BINS * frames)
        val real = FloatArray(FLOW_FFT_PADDED)
        val imag = FloatArray(FLOW_FFT_PADDED)

        var frameIndex = 0
        while (frameIndex < frames) {
            val start = frameIndex * FLOW_HOP

            var i = 0
            while (i < FLOW_N_FFT) {
                real[i] = padded[start + i] * hannWindow1920[i]
                imag[i] = 0f
                i += 1
            }
            while (i < FLOW_FFT_PADDED) {
                real[i] = 0f
                imag[i] = 0f
                i += 1
            }
            fftRadix2InPlace(real, imag)

            var bin = 0
            while (bin < FLOW_FFT_BINS) {
                val re = real[bin]
                val im = imag[bin]
                mag[bin * frames + frameIndex] = kotlin.math.sqrt(re * re + im * im + 1e-9f)
                bin += 1
            }
            frameIndex += 1
        }

        val mel = FloatArray(FLOW_NUM_MELS * frames)
        var m = 0
        while (m < FLOW_NUM_MELS) {
            val melOffset = m * FLOW_FFT_BINS
            frameIndex = 0
            while (frameIndex < frames) {
                var sum = 0f
                var k = 0
                while (k < FLOW_FFT_BINS) {
                    sum += librosaMelFn[melOffset + k] * mag[k * frames + frameIndex]
                    k += 1
                }
                mel[m * frames + frameIndex] = ln(max(sum, FLOW_LOG_CLIP))
                frameIndex += 1
            }
            m += 1
        }

        val speechFeat = FloatArray(frames * FLOW_NUM_MELS)
        frameIndex = 0
        while (frameIndex < frames) {
            m = 0
            while (m < FLOW_NUM_MELS) {
                speechFeat[frameIndex * FLOW_NUM_MELS + m] = mel[m * frames + frameIndex]
                m += 1
            }
            frameIndex += 1
        }

        return speechFeat to frames
    }

    private fun extractKaldiFbank(waveform: FloatArray): Pair<FloatArray, Int> {
        if (waveform.size < KALDI_WINDOW_SIZE) return FloatArray(0) to 0

        val frameCount = (waveform.size - KALDI_WINDOW_SIZE) / KALDI_WINDOW_SHIFT + 1
        if (frameCount <= 0) return FloatArray(0) to 0

        val output = FloatArray(frameCount * KALDI_NUM_MELS)

        val frame = FloatArray(KALDI_WINDOW_SIZE)
        val real = FloatArray(KALDI_PADDED_WINDOW)
        val imag = FloatArray(KALDI_PADDED_WINDOW)
        val power = FloatArray(KALDI_FFT_BINS)

        var frameIndex = 0
        while (frameIndex < frameCount) {
            val start = frameIndex * KALDI_WINDOW_SHIFT
            var i = 0
            while (i < KALDI_WINDOW_SIZE) {
                frame[i] = waveform[start + i]
                i += 1
            }

            var mean = 0f
            i = 0
            while (i < KALDI_WINDOW_SIZE) {
                mean += frame[i]
                i += 1
            }
            mean /= KALDI_WINDOW_SIZE.toFloat()
            i = 0
            while (i < KALDI_WINDOW_SIZE) {
                frame[i] -= mean
                i += 1
            }

            val x0 = frame[0]
            i = KALDI_WINDOW_SIZE - 1
            while (i >= 1) {
                frame[i] -= PREEMPHASIS * frame[i - 1]
                i -= 1
            }
            frame[0] = x0 * (1f - PREEMPHASIS)

            i = 0
            while (i < KALDI_WINDOW_SIZE) {
                frame[i] *= poveyWindow400[i]
                i += 1
            }

            java.util.Arrays.fill(real, 0f)
            java.util.Arrays.fill(imag, 0f)
            System.arraycopy(frame, 0, real, 0, KALDI_WINDOW_SIZE)

            fftRadix2InPlace(real, imag)

            var k = 0
            while (k < KALDI_FFT_BINS) {
                val re = real[k]
                val im = imag[k]
                power[k] = re * re + im * im
                k += 1
            }

            var mel = 0
            while (mel < KALDI_NUM_MELS) {
                val melOffset = mel * KALDI_FFT_BINS
                var sum = 0f
                k = 0
                while (k < KALDI_FFT_BINS) {
                    sum += melBanks80x257[melOffset + k] * power[k]
                    k += 1
                }
                output[frameIndex * KALDI_NUM_MELS + mel] = ln(max(sum, KALDI_EPSILON))
                mel += 1
            }

            frameIndex += 1
        }

        subtractMeanDim0(output, frameCount, KALDI_NUM_MELS)
        return output to frameCount
    }

    private fun subtractMeanDim0(flat: FloatArray, rows: Int, cols: Int) {
        if (rows <= 0 || cols <= 0) return
        val means = FloatArray(cols)

        var r = 0
        while (r < rows) {
            val rowOffset = r * cols
            var c = 0
            while (c < cols) {
                means[c] += flat[rowOffset + c]
                c += 1
            }
            r += 1
        }

        var c = 0
        while (c < cols) {
            means[c] /= rows.toFloat()
            c += 1
        }

        r = 0
        while (r < rows) {
            val rowOffset = r * cols
            c = 0
            while (c < cols) {
                flat[rowOffset + c] -= means[c]
                c += 1
            }
            r += 1
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
            val wLenR = kotlin.math.cos(theta).toFloat()
            val wLenI = kotlin.math.sin(theta).toFloat()

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

    private fun flattenToIntArray(value: Any?): IntArray {
        val out = ArrayList<Int>(256)
        collectInts(value, out)
        return out.toIntArray()
    }

    private fun flattenToFloatArray(value: Any?): FloatArray {
        val out = ArrayList<Float>(256)
        collectFloats(value, out)
        return FloatArray(out.size) { i -> out[i] }
    }

    private fun collectInts(value: Any?, out: MutableList<Int>) {
        when (value) {
            null -> Unit
            is Int -> out.add(value)
            is Long -> out.add(value.toInt())
            is Short -> out.add(value.toInt())
            is Byte -> out.add(value.toInt())
            is IntArray -> value.forEach { out.add(it) }
            is LongArray -> value.forEach { out.add(it.toInt()) }
            is ShortArray -> value.forEach { out.add(it.toInt()) }
            is ByteArray -> value.forEach { out.add(it.toInt()) }
            is Array<*> -> value.forEach { collectInts(it, out) }
            else -> error("Unsupported speech tokenizer output type: ${value::class.java.name}")
        }
    }

    private fun collectFloats(value: Any?, out: MutableList<Float>) {
        when (value) {
            null -> Unit
            is Float -> out.add(value)
            is Double -> out.add(value.toFloat())
            is Int -> out.add(value.toFloat())
            is Long -> out.add(value.toFloat())
            is FloatArray -> value.forEach { out.add(it) }
            is DoubleArray -> value.forEach { out.add(it.toFloat()) }
            is IntArray -> value.forEach { out.add(it.toFloat()) }
            is LongArray -> value.forEach { out.add(it.toFloat()) }
            is Array<*> -> value.forEach { collectFloats(it, out) }
            else -> error("Unsupported campplus output type: ${value::class.java.name}")
        }
    }

    private fun reflectPad(audio: FloatArray, padLeft: Int, padRight: Int): FloatArray {
        val n = audio.size
        if (n < 2) return audio.copyOf()

        val safePadLeft = min(padLeft, n - 1)
        val safePadRight = min(padRight, n - 1)
        val out = FloatArray(n + safePadLeft + safePadRight)

        var i = 0
        while (i < safePadLeft) {
            out[i] = audio[safePadLeft - i]
            i += 1
        }

        System.arraycopy(audio, 0, out, safePadLeft, n)

        i = 0
        while (i < safePadRight) {
            out[safePadLeft + n + i] = audio[n - 2 - i]
            i += 1
        }
        return out
    }

    private fun reflectPadTorch1D(x: FloatArray, pad: Int): FloatArray {
        if (pad <= 0) return x.copyOf()
        if (x.size <= pad) return x.copyOf()

        val out = FloatArray(x.size + 2 * pad)
        var i = 0
        while (i < pad) {
            out[i] = x[pad - i]
            i += 1
        }
        System.arraycopy(x, 0, out, pad, x.size)

        val start = x.size - pad - 1
        i = 0
        while (i < pad) {
            out[pad + x.size + i] = x[start + (pad - 1 - i)]
            i += 1
        }
        return out
    }

    private fun sanitizeAudio(audio: FloatArray): FloatArray {
        if (audio.isEmpty()) return FloatArray(0)
        val out = FloatArray(audio.size)
        var i = 0
        while (i < audio.size) {
            val v = audio[i]
            out[i] = when {
                v.isNaN() || v.isInfinite() -> 0f
                v > 1f -> 1f
                v < -1f -> -1f
                else -> v
            }
            i += 1
        }
        return out
    }

    private fun resampleLinear(input: FloatArray, srcRate: Int, dstRate: Int): FloatArray {
        if (input.isEmpty()) return FloatArray(0)
        if (srcRate <= 0 || dstRate <= 0 || srcRate == dstRate) return input.copyOf()

        val outLength = ((input.size.toDouble() * dstRate.toDouble()) / srcRate.toDouble())
            .toInt()
            .coerceAtLeast(1)
        val output = FloatArray(outLength)
        val ratio = srcRate.toDouble() / dstRate.toDouble()

        var i = 0
        while (i < outLength) {
            val srcPos = i * ratio
            val left = srcPos.toInt().coerceIn(0, input.lastIndex)
            val right = (left + 1).coerceAtMost(input.lastIndex)
            val frac = (srcPos - left).toFloat()
            output[i] = input[left] * (1f - frac) + input[right] * frac
            i += 1
        }
        return output
    }

    private fun loadRawFloatArray(file: File, expectedCount: Int): FloatArray {
        val bytes = file.readBytes()
        require(bytes.size == expectedCount * 4) {
            "Invalid float bin size for ${file.name}: expected=${expectedCount * 4}, actual=${bytes.size}"
        }
        val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val out = FloatArray(expectedCount)
        bb.asFloatBuffer().get(out)
        return out
    }

    companion object {
        private const val TARGET_SAMPLE_RATE = 16_000
        private const val TARGET_RESAMPLE_RATE = 24_000

        private const val N_FFT_TOKEN = 400
        private const val HOP_LENGTH_TOKEN = 160
        private const val N_MELS_TOKEN = 128
        private const val FFT_BINS_TOKEN = N_FFT_TOKEN / 2 + 1

        private const val KALDI_WINDOW_SIZE = 400
        private const val KALDI_WINDOW_SHIFT = 160
        private const val KALDI_PADDED_WINDOW = 512
        private const val KALDI_FFT_BINS = 257
        private const val KALDI_NUM_MELS = 80
        private const val KALDI_EPSILON = 1.1920929e-7f
        private const val PREEMPHASIS = 0.97f

        private const val FLOW_N_FFT = 1920
        private const val FLOW_FFT_PADDED = 2048
        private const val FLOW_HOP = 480
        private const val FLOW_NUM_MELS = 80
        private const val FLOW_FFT_BINS = FLOW_N_FFT / 2 + 1
        private const val FLOW_PAD = (FLOW_N_FFT - FLOW_HOP) / 2
        private const val FLOW_LOG_CLIP = 1e-5f
    }
}
