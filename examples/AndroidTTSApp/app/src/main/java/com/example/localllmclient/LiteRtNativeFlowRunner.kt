package com.example.llama

import java.io.Closeable
import java.io.File

class LiteRtNativeFlowRunner private constructor(
    private var nativeHandle: Long,
    val runtimeMode: String,
) : Closeable {

    fun forwardFromBin(inputDir: File): FloatArray {
        require(nativeHandle != 0L) { "LiteRtNativeFlowRunner has been closed" }
        require(inputDir.exists() && inputDir.isDirectory) {
            "flow input dir not found: ${inputDir.absolutePath}"
        }
        return nativeRunFromBin(nativeHandle, inputDir.absolutePath)
    }

    override fun close() {
        val handle = nativeHandle
        if (handle != 0L) {
            nativeDestroy(handle)
            nativeHandle = 0L
        }
    }

    companion object {
        init {
            System.loadLibrary("ai-chat")
        }

        fun load(modelFile: File): LiteRtNativeFlowRunner {
            require(modelFile.exists() && modelFile.isFile) {
                "LiteRT flow model not found: ${modelFile.absolutePath}"
            }
            val loaded = nativeCreate(modelFile.absolutePath)
            require(loaded.handle != 0L) { "native LiteRT create returned null handle" }
            return LiteRtNativeFlowRunner(
                nativeHandle = loaded.handle,
                runtimeMode = loaded.runtime,
            )
        }

        @JvmStatic
        private external fun nativeCreate(modelPath: String): NativeCreateResult

        @JvmStatic
        private external fun nativeRunFromBin(handle: Long, inputDir: String): FloatArray

        @JvmStatic
        private external fun nativeDestroy(handle: Long)
    }
}

data class NativeCreateResult(
    val handle: Long,
    val runtime: String,
)
