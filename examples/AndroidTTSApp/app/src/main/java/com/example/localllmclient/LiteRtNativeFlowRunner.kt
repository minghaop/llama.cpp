package com.example.llama

import android.content.Context
import android.util.Log
import com.google.ai.edge.litert.BuiltinNpuAcceleratorProvider
import java.io.Closeable
import java.io.File
import kotlinx.coroutines.runBlocking

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
        private const val TAG = "LiteRtNativeFlowRunner"

        init {
            System.loadLibrary("ai-chat")
        }

        fun load(context: Context, modelFile: File): LiteRtNativeFlowRunner {
            require(modelFile.exists() && modelFile.isFile) {
                "LiteRT flow model not found: ${modelFile.absolutePath}"
            }
            val libraryDir = prepareNpuRuntimeLibraryDir(context)
            val loaded = nativeCreate(modelFile.absolutePath, libraryDir)
            require(loaded.handle != 0L) { "native LiteRT create returned null handle" }
            return LiteRtNativeFlowRunner(
                nativeHandle = loaded.handle,
                runtimeMode = loaded.runtime,
            )
        }

        private fun prepareNpuRuntimeLibraryDir(context: Context): String {
            val provider = BuiltinNpuAcceleratorProvider(context)
            val supported = runCatching { provider.isDeviceSupported() }.getOrElse { false }
            Log.i(TAG, "NPU provider supported=$supported")
            val readyBefore = runCatching { provider.isLibraryReady() }.getOrElse { false }
            if (!readyBefore) {
                runCatching {
                    Log.i(TAG, "NPU runtime library not ready, try download.")
                    runBlocking { provider.downloadLibrary() }
                }.onFailure { err ->
                    Log.w(TAG, "NPU runtime download failed: ${err.message}")
                }
            }
            val readyAfter = runCatching { provider.isLibraryReady() }.getOrElse { false }
            val providerDir = runCatching { provider.getLibraryDir().trim() }.getOrDefault("")
            val appNativeLibDir = runCatching { context.applicationInfo.nativeLibraryDir.trim() }.getOrDefault("")
            val libraryDir = providerDir.ifEmpty { appNativeLibDir }
            Log.i(
                TAG,
                "NPU runtime status: readyBefore=$readyBefore readyAfter=$readyAfter providerDir=${providerDir.ifEmpty { "<empty>" }} appNativeLibDir=${appNativeLibDir.ifEmpty { "<empty>" }} libraryDir=${libraryDir.ifEmpty { "<empty>" }}",
            )
            return libraryDir
        }

        @JvmStatic
        private external fun nativeCreate(modelPath: String, libraryDir: String): NativeCreateResult

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
