package com.example.llama

import android.util.Log
import java.io.Closeable
import java.io.File

/**
 * Flow decoder runner backed by MNN native runtime.
 *
 * Supports legacy decoder inputs and prepared estimator-test inputs.
 */
class MnnFlowDecoderRunner private constructor(
    private var nativeHandle: Long,
) : Closeable {

    fun forward(
        mu: FloatArray,
        mask: FloatArray,
        z: FloatArray,
        spks: FloatArray,
        cond: FloatArray,
    ): FloatArray {
        check(nativeHandle != 0L) { "MNN Flow decoder runner has been closed" }
        require(mu.isNotEmpty()) { "Flow decoder input mu is empty" }
        require(mask.isNotEmpty()) { "Flow decoder input mask is empty" }
        require(z.isNotEmpty()) { "Flow decoder input z is empty" }
        require(spks.isNotEmpty()) { "Flow decoder input spks is empty" }
        require(cond.isNotEmpty()) { "Flow decoder input cond is empty" }
        return nativeForward(
            handle = nativeHandle,
            mu = mu,
            mask = mask,
            z = z,
            spks = spks,
            cond = cond,
        )
    }

    fun forwardPrepared6(
        xIn: FloatArray,
        maskIn: FloatArray,
        muIn: FloatArray,
        tIn: FloatArray,
        spksIn: FloatArray,
        condIn: FloatArray,
    ): FloatArray {
        check(nativeHandle != 0L) { "MNN Flow decoder runner has been closed" }
        require(xIn.isNotEmpty()) { "Flow decoder prepared input x_in is empty" }
        require(maskIn.isNotEmpty()) { "Flow decoder prepared input mask_in is empty" }
        require(muIn.isNotEmpty()) { "Flow decoder prepared input mu_in is empty" }
        require(tIn.isNotEmpty()) { "Flow decoder prepared input t_in is empty" }
        require(spksIn.isNotEmpty()) { "Flow decoder prepared input spks_in is empty" }
        require(condIn.isNotEmpty()) { "Flow decoder prepared input cond_in is empty" }
        return nativeForwardPrepared6(
            handle = nativeHandle,
            xIn = xIn,
            maskIn = maskIn,
            muIn = muIn,
            tIn = tIn,
            spksIn = spksIn,
            condIn = condIn,
        )
    }

    fun forwardPrepared8(
        x: FloatArray,
        dt: FloatArray,
        xIn: FloatArray,
        maskIn: FloatArray,
        muIn: FloatArray,
        tIn: FloatArray,
        spksIn: FloatArray,
        condIn: FloatArray,
    ): FloatArray {
        check(nativeHandle != 0L) { "MNN Flow decoder runner has been closed" }
        require(x.isNotEmpty()) { "Flow decoder prepared input x is empty" }
        require(dt.isNotEmpty()) { "Flow decoder prepared input dt is empty" }
        require(xIn.isNotEmpty()) { "Flow decoder prepared input x_in is empty" }
        require(maskIn.isNotEmpty()) { "Flow decoder prepared input mask_in is empty" }
        require(muIn.isNotEmpty()) { "Flow decoder prepared input mu_in is empty" }
        require(tIn.isNotEmpty()) { "Flow decoder prepared input t_in is empty" }
        require(spksIn.isNotEmpty()) { "Flow decoder prepared input spks_in is empty" }
        require(condIn.isNotEmpty()) { "Flow decoder prepared input cond_in is empty" }
        return nativeForwardPrepared8(
            handle = nativeHandle,
            x = x,
            dt = dt,
            xIn = xIn,
            maskIn = maskIn,
            muIn = muIn,
            tIn = tIn,
            spksIn = spksIn,
            condIn = condIn,
        )
    }

    fun consumeLastProfileSummary(): String? {
        check(nativeHandle != 0L) { "MNN Flow decoder runner has been closed" }
        return nativeConsumeLastProfileSummary(nativeHandle)
    }

    override fun close() {
        if (nativeHandle != 0L) {
            nativeUnloadModel(nativeHandle)
            nativeHandle = 0L
        }
    }

    companion object {
        private const val TAG = "MnnFlowDecoderRunner"

        init {
            System.loadLibrary("ai-chat")
        }

        fun load(
            modelFile: File,
            numThreads: Int = 4,
            enableOpProfile: Boolean = false,
        ): MnnFlowDecoderRunner {
            val resolvedModelFile = resolveModelFileCompat(modelFile)
            val handle = nativeLoadModel(
                resolvedModelFile.absolutePath,
                numThreads.coerceAtLeast(1),
                enableOpProfile,
            )
            require(handle != 0L) { "Failed to load MNN flow decoder model: ${resolvedModelFile.absolutePath}" }
            Log.i(TAG, "MNN flow decoder model loaded: ${resolvedModelFile.absolutePath}")
            return MnnFlowDecoderRunner(handle)
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
                            "Flow decoder model file name case mismatch fixed: ${caseInsensitiveMatch.name} -> ${requested.name}",
                        )
                        return requested
                    }
                    Log.w(
                        TAG,
                        "Flow decoder model file name case mismatch: requested=${requested.name}, found=${caseInsensitiveMatch.name}. Using legacy file.",
                    )
                }
                return caseInsensitiveMatch
            }

            error("MNN flow decoder model not found: ${requested.absolutePath}")
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
            mu: FloatArray,
            mask: FloatArray,
            z: FloatArray,
            spks: FloatArray,
            cond: FloatArray,
        ): FloatArray

        @JvmStatic
        private external fun nativeForwardPrepared6(
            handle: Long,
            xIn: FloatArray,
            maskIn: FloatArray,
            muIn: FloatArray,
            tIn: FloatArray,
            spksIn: FloatArray,
            condIn: FloatArray,
        ): FloatArray

        @JvmStatic
        private external fun nativeForwardPrepared8(
            handle: Long,
            x: FloatArray,
            dt: FloatArray,
            xIn: FloatArray,
            maskIn: FloatArray,
            muIn: FloatArray,
            tIn: FloatArray,
            spksIn: FloatArray,
            condIn: FloatArray,
        ): FloatArray

        @JvmStatic
        private external fun nativeConsumeLastProfileSummary(handle: Long): String?
    }
}
