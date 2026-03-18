package com.example.llama

import androidx.annotation.Keep
import java.io.File

@Keep
class QwenTokenizerService private constructor() : AutoCloseable {
    private val lock = Any()

    @Volatile
    private var nativeHandle: Long = 0

    fun encodeSync(text: String): IntArray =
        synchronized(lock) {
            val handle = requireHandle()
            nativeEncode(handle, text)
        }

    fun decodeSync(tokens: IntArray): String =
        synchronized(lock) {
            val handle = requireHandle()
            nativeDecode(handle, tokens)
        }

    override fun close() {
        synchronized(lock) {
            val handle = nativeHandle
            if (handle != 0L) {
                nativeDestroy(handle)
                nativeHandle = 0
            }
        }
    }

    private fun requireHandle(): Long {
        val handle = nativeHandle
        check(handle != 0L) { "QwenTokenizerService is closed or not initialized" }
        return handle
    }

    @Keep
    private external fun nativeLoad(tokenizerJsonPath: String): Long

    @Keep
    private external fun nativeEncode(handle: Long, text: String): IntArray

    @Keep
    private external fun nativeDecode(handle: Long, tokenIds: IntArray): String

    @Keep
    private external fun nativeDestroy(handle: Long)

    companion object {
        private const val TOKENIZER_JSON = "tokenizer.json"

        init {
            runCatching { System.loadLibrary("ai-chat") }
        }

        fun load(from: File): QwenTokenizerService {
            require(from.exists() && from.isDirectory) { "Invalid tokenizer directory: ${from.absolutePath}" }

            val tokenizerJson = File(from, TOKENIZER_JSON)
            require(tokenizerJson.exists() && tokenizerJson.isFile) {
                "Missing tokenizer.json in ${from.absolutePath}"
            }

            val service = QwenTokenizerService()
            val handle = service.nativeLoad(tokenizerJson.absolutePath)
            check(handle != 0L) { "Failed to load tokenizer from ${tokenizerJson.absolutePath}" }
            service.nativeHandle = handle
            return service
        }
    }
}
