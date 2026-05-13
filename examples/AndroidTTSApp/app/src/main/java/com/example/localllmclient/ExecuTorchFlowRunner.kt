package com.example.llama

import android.util.Log
import java.io.Closeable
import java.io.File

/**
 * 通过反射调用 ExecuTorch Android Java API，避免在当前仓库强绑定依赖坐标。
 *
 * 约定 forward 输入顺序：
 * 0: token(int64[])
 * 1: token_len(int64 标量)
 * 2: prompt_token(int64[])
 * 3: prompt_token_len(int64 标量)
 * 4: prompt_feat(float32[])
 * 5: prompt_feat_len(int64 标量)
 * 6: embedding(float32[])
 * 7: streaming(bool)
 * 8: finalize(bool)
 */
class ExecuTorchFlowRunner private constructor(
    private val module: Any,
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
        val tensorCls = Class.forName("org.pytorch.executorch.Tensor")
        val eValueCls = Class.forName("org.pytorch.executorch.EValue")
        val moduleCls = Class.forName("org.pytorch.executorch.Module")

        val fromFloatTensor = tensorCls.getMethod("fromBlob", FloatArray::class.java, LongArray::class.java)
        val fromIntTensor = tensorCls.getMethod("fromBlob", IntArray::class.java, LongArray::class.java)
        val fromLongTensor = tensorCls.getMethod("fromBlob", LongArray::class.java, LongArray::class.java)
        val eFromTensor = eValueCls.getMethod("from", tensorCls)
        val eFromBool = eValueCls.getMethod("from", java.lang.Boolean.TYPE)

        val safeTokenLen = tokenLen.coerceIn(0, token.size)
        val safePromptTokenLen = promptTokenLen.coerceIn(0, promptToken.size)

        val tokenSlice = if (safeTokenLen > 0) token.copyOfRange(0, safeTokenLen) else LongArray(0)
        val promptTokenSlice = if (safePromptTokenLen > 0) promptToken.copyOfRange(0, safePromptTokenLen) else LongArray(0)

        val tokenTensor = fromLongTensor.invoke(null, tokenSlice, longArrayOf(1L, tokenSlice.size.toLong()))
        val tokenLenTensor = fromIntTensor.invoke(null, intArrayOf(safeTokenLen), longArrayOf(1L))
        val promptTokenTensor = fromLongTensor.invoke(null, promptTokenSlice, longArrayOf(1L, promptTokenSlice.size.toLong()))
        val promptTokenLenTensor = fromIntTensor.invoke(null, intArrayOf(safePromptTokenLen), longArrayOf(1L))
        val promptFeatShape = if (promptFeatLen > 0 && promptFeat.size % promptFeatLen == 0) {
            longArrayOf(1L, promptFeatLen.toLong(), (promptFeat.size / promptFeatLen).toLong())
        } else {
            longArrayOf(1L, promptFeat.size.toLong())
        }
        val promptFeatTensor = fromFloatTensor.invoke(null, promptFeat, promptFeatShape)
        val promptFeatLenTensor = fromIntTensor.invoke(null, intArrayOf(promptFeatLen.coerceAtLeast(0)), longArrayOf(1L))
        val embeddingTensor = fromFloatTensor.invoke(null, embedding, longArrayOf(1L, embedding.size.toLong()))

        val inputs = java.lang.reflect.Array.newInstance(eValueCls, 9)
        java.lang.reflect.Array.set(inputs, 0, eFromTensor.invoke(null, tokenTensor))
        java.lang.reflect.Array.set(inputs, 1, eFromTensor.invoke(null, tokenLenTensor))
        java.lang.reflect.Array.set(inputs, 2, eFromTensor.invoke(null, promptTokenTensor))
        java.lang.reflect.Array.set(inputs, 3, eFromTensor.invoke(null, promptTokenLenTensor))
        java.lang.reflect.Array.set(inputs, 4, eFromTensor.invoke(null, promptFeatTensor))
        java.lang.reflect.Array.set(inputs, 5, eFromTensor.invoke(null, promptFeatLenTensor))
        java.lang.reflect.Array.set(inputs, 6, eFromTensor.invoke(null, embeddingTensor))
        java.lang.reflect.Array.set(inputs, 7, eFromBool.invoke(null, streaming))
        java.lang.reflect.Array.set(inputs, 8, eFromBool.invoke(null, finalize))

        val forward = moduleCls.getMethod("forward", inputs.javaClass)
        val outputs = forward.invoke(module, inputs) as Array<*>
        require(outputs.isNotEmpty()) { "ExecuTorch forward returned empty outputs" }

        val first = outputs[0] ?: error("ExecuTorch first output is null")
        val toTensor = eValueCls.getMethod("toTensor")
        val outTensor = toTensor.invoke(first)

        val asFloatArray = tensorCls.getMethod("getDataAsFloatArray")
        @Suppress("UNCHECKED_CAST")
        return asFloatArray.invoke(outTensor) as FloatArray
    }

    override fun close() {
        runCatching {
            val destroy = module.javaClass.getMethod("destroy")
            destroy.invoke(module)
        }
    }

    companion object {
        private const val TAG = "ExecuTorchFlowRunner"

        fun load(modelFile: File, numThreads: Int = 0): ExecuTorchFlowRunner {
            val moduleCls = Class.forName("org.pytorch.executorch.Module")
            val loadMethod = runCatching {
                moduleCls.getMethod(
                    "load",
                    String::class.java,
                    Int::class.javaPrimitiveType,
                    Int::class.javaPrimitiveType,
                )
            }.getOrNull()

            val module = if (loadMethod != null) {
                val loadModeMmap = moduleCls.getField("LOAD_MODE_MMAP").getInt(null)
                loadMethod.invoke(null, modelFile.absolutePath, loadModeMmap, numThreads)
            } else {
                val legacyLoad = moduleCls.getMethod("load", String::class.java)
                legacyLoad.invoke(null, modelFile.absolutePath)
            }

            Log.i(TAG, "ExecuTorch module loaded: ${modelFile.absolutePath}")
            return ExecuTorchFlowRunner(requireNotNull(module) { "ExecuTorch Module.load returned null" })
        }
    }
}
