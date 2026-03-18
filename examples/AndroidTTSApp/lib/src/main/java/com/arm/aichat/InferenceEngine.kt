package com.arm.aichat

import com.arm.aichat.InferenceEngine.State
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.StateFlow

/**
 * Interface defining the core LLM inference operations.
 */
interface InferenceEngine {
    /**
     * Current state of the inference engine
     */
    val state: StateFlow<State>

    /**
     * Load a model from the given path.
     *
     * @throws UnsupportedArchitectureException if model architecture not supported
     */
    suspend fun loadModel(pathToModel: String)

    /**
     * Sends a system prompt to the loaded model
     */
    suspend fun setSystemPrompt(systemPrompt: String)

    /**
     * Sends a user prompt to the loaded model and returns a Flow of generated tokens.
     */
    fun sendUserPrompt(message: String, predictLength: Int = DEFAULT_PREDICT_LENGTH): Flow<String>

    /**
     * Sends pre-tokenized user prompt ids to the loaded model and returns a Flow of generated tokens.
     */
    fun sendUserPrompt(tokenIds: IntArray, predictLength: Int = DEFAULT_PREDICT_LENGTH): Flow<String>

    /**
     * Clears KV-cache and related decode state, preparing for embedding-driven decode loops.
     */
    suspend fun resetKvCache()

    /**
     * Runs one llama_decode pass using input embeddings and returns output embeddings.
     */
    suspend fun decodeEmbeddings(inputEmbeddings: FloatArray, nPast: Int = 0): FloatArray

    /**
     * Loads CosyVoice Flow model and prepares its dedicated context.
     */
    suspend fun loadFlowModel(pathToModel: String)

    /**
     * Loads CosyVoice HIFT model and prepares its dedicated context.
     */
    suspend fun loadHiftModel(pathToModel: String)

    /**
     * Runs one Flow encode pass and returns flattened flow output.
     */
    suspend fun encodeFlow(
        inputEmbeddings: FloatArray,
        flowFeat: FloatArray,
        flowToken: IntArray,
        tokenLen: Int,
        promptTokenLen: Int,
        promptFeatLen: Int,
        randNoise: FloatArray,
        extendPe: FloatArray,
    ): FloatArray

    /**
     * Runs one HIFT encode pass and returns flattened hift output.
     */
    suspend fun encodeHift(inputEmbeddings: FloatArray): FloatArray

    /**
     * Runs a benchmark with the specified parameters.
     */
    suspend fun bench(pp: Int, tg: Int, pl: Int, nr: Int = 1): String

    /**
     * Unloads the currently loaded model.
     */
    fun cleanUp()

    /**
     * Cleans up resources when the engine is no longer needed.
     */
    fun destroy()

    /**
     * States of the inference engine
     */
    sealed class State {
        object Uninitialized : State()
        object Initializing : State()
        object Initialized : State()

        object LoadingModel : State()
        object UnloadingModel : State()
        object ModelReady : State()

        object Benchmarking : State()
        object ProcessingSystemPrompt : State()
        object ProcessingUserPrompt : State()

        object Generating : State()

        data class Error(val exception: Exception) : State()
    }

    companion object {
        const val DEFAULT_PREDICT_LENGTH = 1024
    }
}

val State.isUninterruptible
    get() = this is State.Initializing ||
        this is State.LoadingModel ||
        this is State.UnloadingModel ||
        this is State.Benchmarking ||
        this is State.ProcessingSystemPrompt ||
        this is State.ProcessingUserPrompt

val State.isModelLoaded: Boolean
    get() = this is State.ModelReady ||
        this is State.Benchmarking ||
        this is State.ProcessingSystemPrompt ||
        this is State.ProcessingUserPrompt ||
        this is State.Generating

class UnsupportedArchitectureException : Exception()
