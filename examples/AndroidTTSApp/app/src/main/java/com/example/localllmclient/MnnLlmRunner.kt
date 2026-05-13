package com.example.llama

import android.util.Log
import java.io.Closeable
import java.io.File

/**
 * LLM runner backed by MNN Express module inference.
 *
 * The decode API mirrors InferenceEngine.decodeEmbeddings so existing token-sampling
 * logic can remain unchanged.
 */
class MnnLlmRunner private constructor(
    private var nativeHandle: Long,
) : Closeable {

    fun resetKvCache() {
        check(nativeHandle != 0L) { "MNN LLM runner has been closed" }
        val code = nativeResetKvCache(nativeHandle)
        require(code == 0) { "Failed to reset MNN LLM KV cache: code=$code" }
    }

    fun decodeEmbeddings(
        inputEmbeddings: FloatArray,
        nPast: Int = 0,
    ): FloatArray {
        check(nativeHandle != 0L) { "MNN LLM runner has been closed" }
        require(inputEmbeddings.isNotEmpty()) { "inputEmbeddings cannot be empty" }
        return nativeDecodeEmbeddings(
            handle = nativeHandle,
            inputEmbeddings = inputEmbeddings,
            nPast = nPast.coerceAtLeast(0),
        )
    }

    override fun close() {
        if (nativeHandle != 0L) {
            nativeUnloadModel(nativeHandle)
            nativeHandle = 0L
        }
    }

    companion object {
        private const val TAG = "MnnLlmRunner"

        init {
            System.loadLibrary("ai-chat")
        }

        fun load(configFile: File, numThreads: Int = 4): MnnLlmRunner {
            require(configFile.exists() && configFile.isFile) {
                "MNN LLM config not found: ${configFile.absolutePath}"
            }
            val handle = nativeLoadModel(
                configPath = configFile.absolutePath,
                numThreads = numThreads.coerceAtLeast(1),
            )
            require(handle != 0L) { "Failed to load MNN LLM model from ${configFile.absolutePath}" }
            Log.i(TAG, "MNN LLM loaded: ${configFile.absolutePath}")
            return MnnLlmRunner(handle)
        }

        @JvmStatic
        private external fun nativeLoadModel(configPath: String, numThreads: Int): Long

        @JvmStatic
        private external fun nativeUnloadModel(handle: Long)

        @JvmStatic
        private external fun nativeResetKvCache(handle: Long): Int

        @JvmStatic
        private external fun nativeDecodeEmbeddings(
            handle: Long,
            inputEmbeddings: FloatArray,
            nPast: Int,
        ): FloatArray
    }
}
