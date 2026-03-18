package com.example.llama

import java.io.Closeable
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.exp
import kotlin.math.max

private data class WeightEntry(
    val data: FloatArray,
    val rows: Int,
    val cols: Int,
)

class ModelParameters {
    private val weights = LinkedHashMap<String, WeightEntry>()

    fun loadFromBinary(path: String, name: String) {
        val file = File(path)
        val bytes = file.readBytes()
        require(bytes.size >= 8) { "File too small for header: $path" }

        val header = ByteBuffer.wrap(bytes, 0, 8).order(ByteOrder.LITTLE_ENDIAN)
        val rows = header.int
        val cols = header.int
        require(rows > 0 && cols > 0) { "Invalid shape ($rows, $cols) for $name" }

        val count = rows.toLong() * cols.toLong()
        require(count <= Int.MAX_VALUE.toLong()) { "Weight too large: $name" }
        val expected = 8L + count * 4L
        require(bytes.size >= expected.toInt()) {
            "File size mismatch for $name: expected >= $expected, got ${bytes.size}"
        }

        val dataBuf = ByteBuffer.wrap(bytes, 8, (count * 4L).toInt()).order(ByteOrder.LITTLE_ENDIAN)
        val out = FloatArray(count.toInt())
        dataBuf.asFloatBuffer().get(out)
        weights[name] = WeightEntry(out, rows, cols)
    }

    fun embedding(name: String, tokens: IntArray): FloatArray? {
        val entry = weights[name] ?: return null
        val dim = entry.cols
        val out = FloatArray(tokens.size * dim)
        for (i in tokens.indices) {
            val token = tokens[i]
            if (token < 0 || token >= entry.rows) return null
            System.arraycopy(entry.data, token * dim, out, i * dim, dim)
        }
        return out
    }

    fun getEmbeddingRow(name: String, rowIndex: Int): FloatArray? {
        val entry = weights[name] ?: return null
        if (rowIndex < 0 || rowIndex >= entry.rows) return null
        val start = rowIndex * entry.cols
        return entry.data.copyOfRange(start, start + entry.cols)
    }

    fun computeLogProbabilities(lastHiddenState: FloatArray): FloatArray? {
        val weight = weights["llm_decoder.weight"] ?: return null
        val biasEntry = weights["llm_decoder.bias"] ?: return null
        if (weight.cols != lastHiddenState.size) return null

        val outFeatures = weight.rows
        val bias = if (biasEntry.rows == outFeatures && biasEntry.cols == 1) {
            biasEntry.data
        } else if (biasEntry.rows == 1 && biasEntry.cols == outFeatures) {
            biasEntry.data
        } else if (biasEntry.rows == outFeatures && biasEntry.cols == 0) {
            biasEntry.data
        } else {
            return null
        }
        if (bias.size < outFeatures) return null

        val logits = FloatArray(outFeatures)
        val w = weight.data
        val inFeatures = weight.cols
        for (i in 0 until outFeatures) {
            val rowOffset = i * inFeatures
            var dot = 0.0f
            for (j in 0 until inFeatures) {
                dot += lastHiddenState[j] * w[rowOffset + j]
            }
            logits[i] = dot + bias[i]
        }
        return logSoftmax(logits)
    }

    private fun logSoftmax(input: FloatArray): FloatArray {
        if (input.isEmpty()) return FloatArray(0)
        var maxVal = Float.NEGATIVE_INFINITY
        for (v in input) {
            if (v > maxVal) maxVal = v
        }
        var sumExp = 0.0
        for (v in input) {
            sumExp += exp((v - maxVal).toDouble())
        }
        val logSumExp = kotlin.math.ln(sumExp).toFloat()
        return FloatArray(input.size) { i -> (input[i] - maxVal) - logSumExp }
    }

    fun clear() {
        weights.clear()
    }
}

class SeededRNG(seed: Long) {
    private var state: Long = if (seed == 0L) 0x9E3779B97F4A7C15uL.toLong() else seed

    private fun nextUInt64(): Long {
        state += 0x9E3779B97F4A7C15uL.toLong()
        var z = state
        z = (z xor (z ushr 30)) * 0xBF58476D1CE4E5B9uL.toLong()
        z = (z xor (z ushr 27)) * 0x94D049BB133111EBuL.toLong()
        return z xor (z ushr 31)
    }

    fun nextFloat01(): Float {
        val x = nextUInt64() ushr 40
        return (x.toDouble() / (1 shl 24).toDouble()).toFloat()
    }
}

object SamplingAlgorithm {
    const val SPEECH_TOKEN_SIZE = 6561

    fun samplingIds(
        logProbs: FloatArray,
        decoderTokens: IntArray,
        samplingNum: Int,
        ignoreEOS: Boolean,
        rng: SeededRNG,
    ): Int {
        var trials = 0
        val maxTrials = 100
        while (true) {
            val topId = sampling(
                logProbs = logProbs,
                decodedTokens = decoderTokens,
                sampling = samplingNum,
                rng = rng,
            )
            if (!ignoreEOS || topId != SPEECH_TOKEN_SIZE) return topId
            trials++
            if (trials > maxTrials) return topId
        }
    }

    private fun sampling(
        logProbs: FloatArray,
        decodedTokens: IntArray,
        sampling: Int,
        topP: Float = 0.8f,
        topK: Int = 25,
        winSize: Int = 10,
        tauR: Float = 0.1f,
        rng: SeededRNG,
    ): Int {
        var topId = nucleusSampling(logProbs, topP, topK, rng)
        val start = max(0, decodedTokens.size - winSize)
        var repNum = 0
        for (i in start until decodedTokens.size) {
            if (decodedTokens[i] == topId) repNum++
        }
        if (repNum.toFloat() >= winSize.toFloat() * tauR) {
            topId = randomSampling(logProbs, rng)
        }
        return topId
    }

    private fun nucleusSampling(
        logProbs: FloatArray,
        topP: Float,
        topK: Int,
        rng: SeededRNG,
    ): Int {
        if (logProbs.isEmpty()) return 0
        val idx = logProbs.indices.sortedByDescending { logProbs[it] }

        val filteredIndices = ArrayList<Int>(topK)
        val filteredProbs = ArrayList<Float>(topK)
        var cumProb = 0.0f
        for (id in idx) {
            val p = exp(logProbs[id].toDouble()).toFloat()
            if (cumProb < topP && filteredProbs.size < topK) {
                cumProb += p
                filteredIndices.add(id)
                filteredProbs.add(p)
            } else {
                break
            }
        }
        if (filteredIndices.isEmpty()) return idx.first()
        if (filteredIndices.size == 1) return filteredIndices[0]

        var sum = 0.0f
        for (p in filteredProbs) sum += p
        if (sum <= 0) return filteredIndices[0]

        val r = rng.nextFloat01()
        var acc = 0.0f
        for (i in filteredProbs.indices) {
            acc += filteredProbs[i] / sum
            if (r <= acc) return filteredIndices[i]
        }
        return filteredIndices.last()
    }

    private fun randomSampling(logProbs: FloatArray, rng: SeededRNG): Int {
        if (logProbs.isEmpty()) return 0
        val probs = FloatArray(logProbs.size) { i -> exp(logProbs[i].toDouble()).toFloat() }
        val r = rng.nextFloat01()
        var acc = 0.0f
        for (i in probs.indices) {
            acc += probs[i]
            if (r <= acc) return i
        }
        return probs.lastIndex
    }
}

class MMapEmbeddingF32(path: String) : Closeable {
    val rows: Int
    val cols: Int

    private val raf: RandomAccessFile = RandomAccessFile(path, "r")
    private val channel = raf.channel
    private val map = channel.map(java.nio.channels.FileChannel.MapMode.READ_ONLY, 0, channel.size())
    private val floatView: FloatBuffer

    init {
        require(channel.size() >= 8) { "Invalid embedding file (too small): $path" }
        val header = map.duplicate().order(ByteOrder.LITTLE_ENDIAN)
        rows = header.getInt(0)
        cols = header.getInt(4)
        require(rows > 0 && cols > 0) { "Invalid embedding shape: ($rows, $cols)" }
        val expected = 8L + rows.toLong() * cols.toLong() * 4L
        require(channel.size() >= expected) { "Invalid embedding file size: ${channel.size()} < $expected" }

        val payload = map.duplicate().order(ByteOrder.LITTLE_ENDIAN)
        payload.position(8)
        floatView = payload.slice().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
    }

    fun gather(tokens: IntArray): FloatArray {
        val dim = cols
        val out = FloatArray(tokens.size * dim)
        for (i in tokens.indices) {
            val t = tokens[i]
            require(t in 0 until rows) { "Token out of range: $t" }
            val srcOffset = t * dim
            val dup = floatView.duplicate()
            dup.position(srcOffset)
            dup.get(out, i * dim, dim)
        }
        return out
    }

    override fun close() {
        channel.close()
        raf.close()
    }
}

fun embedTokens(embedBinPath: String, tokens: IntArray, expectedDim: Int = 896): FloatArray {
    MMapEmbeddingF32(embedBinPath).use { table ->
        require(table.cols == expectedDim) {
            "Embedding dim mismatch: expected=$expectedDim, got=${table.cols}"
        }
        return table.gather(tokens)
    }
}

fun loadCountPrefixedFloatArray(path: String): FloatArray {
    val bytes = File(path).readBytes()
    require(bytes.size >= 4) { "File too small: $path" }
    val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
    val count = bb.int
    require(count >= 0) { "Invalid float count in $path" }
    val expected = 4 + count * 4
    require(bytes.size >= expected) { "File size mismatch for $path" }
    val out = FloatArray(count)
    for (i in 0 until count) {
        out[i] = bb.float
    }
    return out
}

fun concatFloatArrays(vararg arrays: FloatArray): FloatArray {
    var total = 0
    for (arr in arrays) total += arr.size
    val out = FloatArray(total)
    var cursor = 0
    for (arr in arrays) {
        if (arr.isNotEmpty()) {
            System.arraycopy(arr, 0, out, cursor, arr.size)
            cursor += arr.size
        }
    }
    return out
}
