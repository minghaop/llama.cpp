package com.example.llama

import android.util.Log
import java.io.Closeable
import java.io.File

/**
 * Flow runner backed by MNN native runtime.
 *
 * Input order follows the existing flow_inputs_bin convention:
 * 0: token(int64[])
 * 1: token_len(int32 scalar)
 * 2: prompt_token(int64[])
 * 3: prompt_token_len(int32 scalar)
 * 4: prompt_feat(float32[])
 * 5: prompt_feat_len(int32 scalar)
 * 6: embedding(float32[])
 * 7: streaming(bool)
 * 8: finalize(bool)
 */
class MnnFlowRunner private constructor(
    private var nativeHandle: Long,
) : Closeable {

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
        check(nativeHandle != 0L) { "MNN Flow runner has been closed" }
        return nativeForward(
            handle = nativeHandle,
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
    }

    fun consumeLastProfileSummary(): String? {
        check(nativeHandle != 0L) { "MNN Flow runner has been closed" }
        return nativeConsumeLastProfileSummary(nativeHandle)
    }

    override fun close() {
        if (nativeHandle != 0L) {
            nativeUnloadModel(nativeHandle)
            nativeHandle = 0L
        }
    }

    companion object {
        private const val TAG = "MnnFlowRunner"

        init {
            System.loadLibrary("ai-chat")
        }

        fun load(
            modelFile: File,
            numThreads: Int = 4,
            enableOpProfile: Boolean = false,
        ): MnnFlowRunner {
            val resolvedModelFile = resolveModelFileCompat(modelFile)
            val handle = nativeLoadModel(
                resolvedModelFile.absolutePath,
                numThreads.coerceAtLeast(1),
                enableOpProfile,
            )
            require(handle != 0L) { "Failed to load MNN flow model: ${resolvedModelFile.absolutePath}" }
            Log.i(TAG, "MNN flow model loaded: ${resolvedModelFile.absolutePath}")
            return MnnFlowRunner(handle)
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
                            "Flow model file name case mismatch fixed: ${caseInsensitiveMatch.name} -> ${requested.name}",
                        )
                        return requested
                    }
                    Log.w(
                        TAG,
                        "Flow model file name case mismatch: requested=${requested.name}, found=${caseInsensitiveMatch.name}. Using legacy file.",
                    )
                }
                return caseInsensitiveMatch
            }

            error("MNN flow model not found: ${requested.absolutePath}")
        }

        @JvmStatic
        private external fun nativeLoadModel(
            modelPath: String,
            numThreads: Int,
            enableOpProfile: Boolean,
        ): Long

        @JvmStatic
        private external fun nativeUnloadModel(handle: Long)

        @JvmStatic
        private external fun nativeForward(
            handle: Long,
            token: LongArray,
            tokenLen: Int,
            promptToken: LongArray,
            promptTokenLen: Int,
            promptFeat: FloatArray,
            promptFeatLen: Int,
            embedding: FloatArray,
            streaming: Boolean,
            finalize: Boolean,
        ): FloatArray

        @JvmStatic
        private external fun nativeConsumeLastProfileSummary(handle: Long): String?
    }
}
