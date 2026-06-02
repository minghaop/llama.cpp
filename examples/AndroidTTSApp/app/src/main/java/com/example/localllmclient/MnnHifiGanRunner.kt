package com.example.llama

import android.util.Log
import java.io.Closeable
import java.io.File

/**
 * HifiGan runner backed by MNN native runtime.
 *
 * Typical input shape: [1, 80, T] where T is frame count.
 */
class MnnHifiGanRunner private constructor(
    private var nativeHandle: Long,
) : Closeable {

    fun forward(
        input: FloatArray,
        shape: IntArray,
        outputIndex: Int = 0,
    ): FloatArray {
        check(nativeHandle != 0L) { "MNN HifiGan runner has been closed" }
        require(input.isNotEmpty()) { "HifiGan input is empty" }
        require(shape.isNotEmpty()) { "HifiGan input shape is empty" }
        val expected = shape.fold(1L) { acc, dim ->
            require(dim > 0) { "Invalid HifiGan shape dim: $dim" }
            acc * dim.toLong()
        }
        require(expected == input.size.toLong()) {
            "HifiGan input size mismatch: input=${input.size}, shape=${shape.joinToString("x")} -> $expected"
        }
        return nativeForward(
            handle = nativeHandle,
            input = input,
            shape = shape,
            outputIndex = outputIndex.coerceAtLeast(0),
        )
    }

    override fun close() {
        if (nativeHandle != 0L) {
            nativeUnloadModel(nativeHandle)
            nativeHandle = 0L
        }
    }

    companion object {
        private const val TAG = "MnnHifiGanRunner"
        private const val BACKEND_LABEL = "CPU"

        init {
            runCatching { System.loadLibrary("MNN") }
                .onSuccess { Log.i(TAG, "Loaded libMNN.so") }
                .onFailure { Log.w(TAG, "Unable to preload libMNN.so: ${it.message}") }
            if (BACKEND_LABEL == "Vulkan") {
                runCatching { System.loadLibrary("MNN_Vulkan") }
                    .onSuccess { Log.i(TAG, "Loaded libMNN_Vulkan.so") }
                    .onFailure { Log.w(TAG, "Unable to preload libMNN_Vulkan.so: ${it.message}") }
            }
            System.loadLibrary("ai-chat")
        }

        fun load(modelFile: File, numThreads: Int = 4): MnnHifiGanRunner {
            val resolvedModelFile = resolveModelFileCompat(modelFile)
            val handle = nativeLoadModel(resolvedModelFile.absolutePath, numThreads.coerceAtLeast(1))
            require(handle != 0L) { "Failed to load MNN hifigan model: ${resolvedModelFile.absolutePath}" }
            Log.i(TAG, "MNN hifigan model loaded: ${resolvedModelFile.absolutePath}")
            return MnnHifiGanRunner(handle)
        }

        private fun resolveModelFileCompat(requested: File): File {
            if (requested.exists() && requested.isFile) return requested

            val parent = requested.parentFile
            val caseInsensitiveMatch = parent
                ?.listFiles()
                ?.firstOrNull { it.isFile && it.name.equals(requested.name, ignoreCase = true) }

            if (caseInsensitiveMatch != null) {
                if (caseInsensitiveMatch.name != requested.name) {
                    val migrateOk = caseInsensitiveMatch.renameTo(requested)
                    if (migrateOk && requested.exists() && requested.isFile) {
                        Log.w(
                            TAG,
                            "HifiGan model file name case mismatch fixed: ${caseInsensitiveMatch.name} -> ${requested.name}",
                        )
                        return requested
                    }
                    Log.w(
                        TAG,
                        "HifiGan model file name case mismatch: requested=${requested.name}, found=${caseInsensitiveMatch.name}. Using legacy file.",
                    )
                }
                return caseInsensitiveMatch
            }

            error("MNN hifigan model not found: ${requested.absolutePath}")
        }

        @JvmStatic
        private external fun nativeLoadModel(modelPath: String, numThreads: Int): Long

        @JvmStatic
        private external fun nativeUnloadModel(handle: Long)

        @JvmStatic
        private external fun nativeForward(
            handle: Long,
            input: FloatArray,
            shape: IntArray,
            outputIndex: Int,
        ): FloatArray
    }
}
