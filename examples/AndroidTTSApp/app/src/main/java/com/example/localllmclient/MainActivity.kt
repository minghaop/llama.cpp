package com.example.llama

import android.Manifest
import android.content.Intent
import android.graphics.Color
import android.graphics.drawable.GradientDrawable
import android.media.MediaMetadataRetriever
import android.media.MediaPlayer
import android.graphics.Rect
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.util.TypedValue
import android.view.MotionEvent
import android.view.inputmethod.InputMethodManager
import android.widget.Button
import android.widget.EditText
import android.widget.PopupMenu
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.addCallback
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.widget.doAfterTextChanged
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.ArrayDeque
import java.util.Locale
import java.util.UUID

class MainActivity : AppCompatActivity() {

    private lateinit var inferenceInfoTv: TextView
    private lateinit var promptRecordButton: Button
    private lateinit var promptHistoryButton: Button
    private lateinit var promptTextInput: EditText
    private lateinit var ttsRecordButton: Button
    private lateinit var ttsHistoryButton: Button
    private lateinit var ttsTextInput: EditText
    private lateinit var generateButton: Button
    private lateinit var generatedHistoryButton: Button
    private lateinit var playHistoryButton: Button

    private lateinit var inferenceBridge: LlamaInferenceBridge

    private val promptHistory = mutableListOf<PromptHistoryItem>()
    private val ttsTextHistory = mutableListOf<TTSTextHistoryItem>()
    private val generatedHistory = mutableListOf<GeneratedHistoryItem>()
    private var selectedPromptHistoryId: String? = null
    private var selectedTTSTextHistoryId: String? = null
    private var selectedGeneratedHistoryId: String? = null
    private var nextPromptIndex = 0
    private var nextTTSIndex = 0
    private var nextGeneratedIndex = 0

    private var mediaPlayer: MediaPlayer? = null
    private var historyPlaybackState = HistoryPlaybackState.Stopped

    private var speechRecognizer: SpeechRecognizer? = null
    private val mainHandler = Handler(Looper.getMainLooper())
    private var captureState: CaptureState = CaptureState.Idle
    private var captureSessionId: String? = null

    private var generationJob: Job? = null
    private var generationSeq = 0
    private var isGenerating = false
    private val inferencePromptSampleRate = 16_000

    private val stageEndedInfo = mutableMapOf<InferenceStage, StageEndedDisplay>()
    private val stageRunning = mutableSetOf<InferenceStage>()
    private val stageLastUnitsDone = mutableMapOf<InferenceStage, Int>()
    private val stageLastAvgUPS = mutableMapOf<InferenceStage, Double>()
    private var audioDurationSeconds: Double? = null
    private val eventLogLines = ArrayDeque<String>()
    private var lastProgressRenderMs = 0L

    private val stageOrder = listOf(
        InferenceStage.frontEnd,
        InferenceStage.llmPrepare,
        InferenceStage.llm,
        InferenceStage.flow,
        InferenceStage.hift,
        InferenceStage.voiceGeneration,
    )

    private val requestMicPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { granted ->
        if (!granted) {
            toast("需要麦克风权限才能录音和语音输入")
        }
        refreshControlsEnabled()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)
        configureSystemBars()
        applyWindowInsets()
        onBackPressedDispatcher.addCallback { /* keep simple and consistent with demo app */ }

        bindViews()
        applyStaticStyles()
        installPresets()
        setupSpeechRecognizer()
        setupClicks()
        setupTextSync()
        requestAudioPermissionIfNeeded()

        resetInferenceInfo("请语音输入 Prompt 与 TTS 文本，然后点击「生成」。")
        inferenceBridge = LlamaInferenceBridge(applicationContext)
        refreshControlsEnabled()
    }

    override fun dispatchTouchEvent(ev: MotionEvent): Boolean {
        if (ev.action == MotionEvent.ACTION_DOWN) {
            val focused = currentFocus
            if (focused is EditText) {
                val outRect = Rect()
                focused.getGlobalVisibleRect(outRect)
                if (!outRect.contains(ev.rawX.toInt(), ev.rawY.toInt())) {
                    hideKeyboardAndClearFocus(focused)
                }
            }
        }
        return super.dispatchTouchEvent(ev)
    }

    private fun bindViews() {
        inferenceInfoTv = findViewById(R.id.inference_info_view)
        promptRecordButton = findViewById(R.id.prompt_record_button)
        promptHistoryButton = findViewById(R.id.prompt_history_button)
        promptTextInput = findViewById(R.id.prompt_text_input)
        ttsRecordButton = findViewById(R.id.tts_record_button)
        ttsHistoryButton = findViewById(R.id.tts_history_button)
        ttsTextInput = findViewById(R.id.tts_text_input)
        generateButton = findViewById(R.id.generate_button)
        generatedHistoryButton = findViewById(R.id.generated_history_button)
        playHistoryButton = findViewById(R.id.play_history_button)
    }

    private fun applyWindowInsets() {
        val root = findViewById<ScrollView>(R.id.root_scroll)
        val baseTop = root.paddingTop
        val baseBottom = root.paddingBottom
        val baseLeft = root.paddingLeft
        val baseRight = root.paddingRight
        ViewCompat.setOnApplyWindowInsetsListener(root) { view, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            view.setPadding(
                baseLeft + bars.left,
                baseTop + bars.top,
                baseRight + bars.right,
                baseBottom + bars.bottom,
            )
            insets
        }
        ViewCompat.requestApplyInsets(root)
    }

    private fun configureSystemBars() {
        window.statusBarColor = Color.TRANSPARENT
        val controller = WindowCompat.getInsetsController(window, window.decorView)
        controller.isAppearanceLightStatusBars = true
        controller.isAppearanceLightNavigationBars = true
    }

    private fun applyStaticStyles() {
        promptTextInput.background = roundedRect(COLOR_SYSTEM_GRAY6)
        ttsTextInput.background = roundedRect(COLOR_SYSTEM_GRAY6)
        inferenceInfoTv.background = roundedRect(COLOR_SYSTEM_GRAY5)

        styleNeutralButton(promptHistoryButton)
        styleNeutralButton(ttsHistoryButton)
        stylePrimaryButton(generateButton)
        styleNeutralButton(generatedHistoryButton)
        stylePlayButton()

        promptTextInput.setTextColor(COLOR_LABEL)
        ttsTextInput.setTextColor(COLOR_LABEL)
        inferenceInfoTv.setTextColor(COLOR_LABEL)
    }

    private fun setupClicks() {
        promptRecordButton.setOnClickListener {
            when (val state = captureState) {
                CaptureState.Idle -> startPromptRecording()
                is CaptureState.Recording -> {
                    if (state.target == CaptureTarget.Prompt) {
                        stopCurrentCaptureAndWaitFinal()
                    }
                }
                is CaptureState.RecognizingFinal -> Unit
            }
        }

        ttsRecordButton.setOnClickListener {
            when (val state = captureState) {
                CaptureState.Idle -> startTTSTextRecognition()
                is CaptureState.Recording -> {
                    if (state.target == CaptureTarget.TtsText) {
                        stopCurrentCaptureAndWaitFinal()
                    }
                }
                is CaptureState.RecognizingFinal -> Unit
            }
        }

        promptHistoryButton.setOnClickListener { showPromptHistoryMenu() }
        ttsHistoryButton.setOnClickListener { showTTSTextHistoryMenu() }
        generatedHistoryButton.setOnClickListener { showGeneratedHistoryMenu() }
        generateButton.setOnClickListener { generateTapped() }
        playHistoryButton.setOnClickListener { historyPlayTapped() }
    }

    private fun setupTextSync() {
        promptTextInput.doAfterTextChanged {
            syncSelectedPromptHistoryFromTextView()
            refreshControlsEnabled()
        }
        ttsTextInput.doAfterTextChanged {
            syncSelectedTTSTextHistoryFromTextView()
            refreshControlsEnabled()
        }
    }

    private fun setupSpeechRecognizer() {
        if (!SpeechRecognizer.isRecognitionAvailable(this)) {
            appendEventLine("设备不支持系统语音识别")
            return
        }

        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this).apply {
            setRecognitionListener(object : RecognitionListener {
                override fun onReadyForSpeech(params: Bundle?) = Unit
                override fun onBeginningOfSpeech() = Unit
                override fun onRmsChanged(rmsdB: Float) = Unit
                override fun onBufferReceived(buffer: ByteArray?) = Unit
                override fun onEndOfSpeech() = Unit

                override fun onError(error: Int) {
                    val current = captureState
                    when (current) {
                        is CaptureState.Recording -> {
                            appendEventLine("语音识别错误: $error")
                            captureState = CaptureState.RecognizingFinal(current.target, current.sessionId)
                            finalizeCapture(current.sessionId, current.target)
                        }
                        is CaptureState.RecognizingFinal -> {
                            finalizeCapture(current.sessionId, current.target)
                        }
                        CaptureState.Idle -> {
                            appendEventLine("语音识别错误: $error")
                        }
                    }
                }

                override fun onResults(results: Bundle?) {
                    val text = resultsToText(results)
                    val current = captureState
                    if (text != null && current is CaptureState.RecognizingFinal) {
                        applyRecognizedText(current.target, text)
                    }
                    if (current is CaptureState.RecognizingFinal) {
                        finalizeCapture(current.sessionId, current.target)
                    }
                }

                override fun onPartialResults(partialResults: Bundle?) {
                    val text = resultsToText(partialResults) ?: return
                    when (val current = captureState) {
                        is CaptureState.Recording -> applyRecognizedText(current.target, text)
                        is CaptureState.RecognizingFinal -> applyRecognizedText(current.target, text)
                        CaptureState.Idle -> Unit
                    }
                }

                override fun onEvent(eventType: Int, params: Bundle?) = Unit
            })
        }
    }

    private fun requestAudioPermissionIfNeeded() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) !=
            android.content.pm.PackageManager.PERMISSION_GRANTED
        ) {
            requestMicPermission.launch(Manifest.permission.RECORD_AUDIO)
        }
    }

    private fun installPresets() {
        if (promptHistory.isEmpty()) {
            PRESET_PROMPT_TEXTS.forEachIndexed { index, text ->
                nextPromptIndex += 1
                promptHistory.add(
                    PromptHistoryItem(
                        id = UUID.randomUUID().toString(),
                        index = nextPromptIndex,
                        text = text,
                        audioPath = installPresetPromptAudio(index = index + 1),
                    ),
                )
            }
            promptHistory.firstOrNull()?.let { first ->
                selectedPromptHistoryId = first.id
                promptTextInput.setText(first.text)
            }
        }

        if (ttsTextHistory.isEmpty()) {
            addTTSTextHistoryIfNeeded("突然，身边一阵笑声。我看着他们，意气风发地挺直了胸膛，甩了甩那稍显肉感的双臂，轻笑道：我身上的肉，是为了掩饰我爆棚的魅力，否则，岂不吓坏了你们呢？", select = true)
            addTTSTextHistoryIfNeeded("梯度是一个多变量微积分中的概念，用于描述一个标量场在某一点处的最大变化率，以及变化最快的方向。在物理学中，梯度通常用来表示某个物理量的空间变化情况。", select = false)
        }
    }

    private fun startPromptRecording() {
        if (!hasRecordPermission()) {
            requestMicPermission.launch(Manifest.permission.RECORD_AUDIO)
            return
        }
        if (isGenerating || historyPlaybackState == HistoryPlaybackState.Playing || captureState != CaptureState.Idle) return

        val sessionId = UUID.randomUUID().toString()
        runCatching {
            startSpeechListening(sessionId = sessionId, target = CaptureTarget.Prompt)
            captureSessionId = sessionId
            captureState = CaptureState.Recording(CaptureTarget.Prompt, sessionId)
            appendEventLine("开始语音输入 Prompt 文本")
            refreshControlsEnabled()
        }.onFailure { t ->
            appendEventLine("Prompt 语音输入启动失败: ${t.message}")
            captureState = CaptureState.Idle
            refreshControlsEnabled()
        }
    }

    private fun startTTSTextRecognition() {
        if (!hasRecordPermission()) {
            requestMicPermission.launch(Manifest.permission.RECORD_AUDIO)
            return
        }
        if (isGenerating || historyPlaybackState == HistoryPlaybackState.Playing || captureState != CaptureState.Idle) return

        selectedTTSTextHistoryId = null
        ttsTextInput.setText("")
        refreshTTSHistoryUI()

        val sessionId = UUID.randomUUID().toString()
        runCatching {
            startSpeechListening(sessionId = sessionId, target = CaptureTarget.TtsText)
            captureSessionId = sessionId
            captureState = CaptureState.Recording(CaptureTarget.TtsText, sessionId)
            appendEventLine("开始语音输入 TTS 文本")
            refreshControlsEnabled()
        }.onFailure { t ->
            appendEventLine("语音输入启动失败: ${t.message}")
            captureState = CaptureState.Idle
            refreshControlsEnabled()
        }
    }

    private fun startSpeechListening(sessionId: String, target: CaptureTarget) {
        val recognizer = speechRecognizer ?: error("SpeechRecognizer unavailable")
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.CHINA.toLanguageTag())
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
            putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
        }
        captureSessionId = sessionId
        captureState = CaptureState.Recording(target, sessionId)
        recognizer.startListening(intent)
    }

    private fun stopCurrentCaptureAndWaitFinal() {
        val state = captureState as? CaptureState.Recording ?: return
        captureState = CaptureState.RecognizingFinal(state.target, state.sessionId)
        speechRecognizer?.stopListening()

        val waitMsg = when (state.target) {
            CaptureTarget.Prompt -> "停止语音输入 Prompt，等待识别 final..."
            CaptureTarget.TtsText -> "停止语音输入，等待识别 final..."
        }
        appendEventLine(waitMsg)
        refreshControlsEnabled()

        mainHandler.postDelayed({
            val waiting = captureState as? CaptureState.RecognizingFinal ?: return@postDelayed
            finalizeCapture(waiting.sessionId, waiting.target)
        }, 5_000)
    }

    private fun finalizeCapture(sessionId: String, target: CaptureTarget) {
        val current = captureState as? CaptureState.RecognizingFinal ?: return
        if (current.sessionId != sessionId || current.target != target) return

        when (target) {
            CaptureTarget.Prompt -> {
                val promptText = promptTextInput.text.toString()
                val sid = selectedPromptHistoryId
                if (sid != null) {
                    syncSelectedPromptHistoryFromTextView()
                } else {
                    addPromptHistoryIfNeeded(promptText, select = true, audioPath = null)
                }
                appendEventLine("Prompt 文本识别完成")
            }
            CaptureTarget.TtsText -> {
                val text = ttsTextInput.text.toString()
                addTTSTextHistoryIfNeeded(text, select = true)
                appendEventLine("TTS 文本识别完成")
            }
        }

        captureState = CaptureState.Idle
        captureSessionId = null
        refreshControlsEnabled()
    }

    private fun addPromptHistoryIfNeeded(text: String, select: Boolean, audioPath: String? = null) {
        val trimmed = text.trim()
        if (trimmed.isEmpty()) return
        val first = promptHistory.firstOrNull()
        if (first?.text?.trim() == trimmed) {
            if (select) selectedPromptHistoryId = first.id
            refreshPromptHistoryUI()
            return
        }

        nextPromptIndex += 1
        val item = PromptHistoryItem(
            id = UUID.randomUUID().toString(),
            index = nextPromptIndex,
            text = trimmed,
            audioPath = audioPath,
        )
        promptHistory.add(0, item)
        if (promptHistory.size > MAX_PROMPT_HISTORY) {
            promptHistory.removeAt(promptHistory.lastIndex)
        }
        if (select) {
            selectedPromptHistoryId = item.id
        }
        refreshPromptHistoryUI()
        if (select) {
            promptTextInput.setText(item.text)
        }
    }

    private fun addTTSTextHistoryIfNeeded(text: String, select: Boolean) {
        val trimmed = text.trim()
        if (trimmed.isEmpty()) return
        val first = ttsTextHistory.firstOrNull()
        if (first?.text?.trim() == trimmed) {
            if (select) selectedTTSTextHistoryId = first.id
            refreshTTSHistoryUI()
            return
        }

        nextTTSIndex += 1
        val item = TTSTextHistoryItem(
            id = UUID.randomUUID().toString(),
            index = nextTTSIndex,
            text = trimmed,
        )
        ttsTextHistory.add(0, item)
        if (ttsTextHistory.size > MAX_TTS_HISTORY) {
            ttsTextHistory.removeAt(ttsTextHistory.lastIndex)
        }
        if (select) {
            selectedTTSTextHistoryId = item.id
            ttsTextInput.setText(item.text)
        }
        refreshTTSHistoryUI()
    }

    private fun showPromptHistoryMenu() {
        val popup = PopupMenu(this, promptHistoryButton)
        if (promptHistory.isEmpty()) {
            popup.menu.add("Empty")
        } else {
            promptHistory.forEachIndexed { index, item ->
                popup.menu.add(0, index, index, promptHistoryLabel(item))
            }
        }
        popup.setOnMenuItemClickListener { menuItem ->
            val idx = menuItem.itemId
            if (idx in promptHistory.indices) {
                val item = promptHistory[idx]
                selectedPromptHistoryId = item.id
                promptTextInput.setText(item.text)
                refreshPromptHistoryUI()
            }
            true
        }
        popup.show()
    }

    private fun showTTSTextHistoryMenu() {
        val popup = PopupMenu(this, ttsHistoryButton)
        if (ttsTextHistory.isEmpty()) {
            popup.menu.add("暂无 TTS 文本历史")
        } else {
            ttsTextHistory.forEachIndexed { index, item ->
                popup.menu.add(0, index, index, ttsTextHistoryLabel(item))
            }
        }
        popup.setOnMenuItemClickListener { menuItem ->
            val idx = menuItem.itemId
            if (idx in ttsTextHistory.indices) {
                val item = ttsTextHistory[idx]
                selectedTTSTextHistoryId = item.id
                ttsTextInput.setText(item.text)
                refreshTTSHistoryUI()
            }
            true
        }
        popup.show()
    }

    private fun showGeneratedHistoryMenu() {
        val popup = PopupMenu(this, generatedHistoryButton)
        if (generatedHistory.isEmpty()) {
            popup.menu.add("暂无历史")
        } else {
            generatedHistory.forEachIndexed { index, item ->
                popup.menu.add(0, index, index, generatedHistoryLabel(item))
            }
        }
        popup.setOnMenuItemClickListener { menuItem ->
            val idx = menuItem.itemId
            if (idx in generatedHistory.indices) {
                selectedGeneratedHistoryId = generatedHistory[idx].id
                refreshGeneratedHistoryUI()
            }
            true
        }
        popup.show()
    }

    private fun generateTapped() {
        hideKeyboardAndClearFocus(currentFocus)
        if (isGenerating) return
        if (captureState != CaptureState.Idle) {
            appendEventLine("录音/识别进行中，请稍后再生成")
            return
        }
        if (historyPlaybackState == HistoryPlaybackState.Playing) {
            appendEventLine("播放中不能生成，请先停止播放")
            return
        }

        val promptText = promptTextInput.text.toString().trim()
        if (promptText.isEmpty()) {
            appendEventLine("Prompt 文本为空")
            return
        }
        if (selectedPromptHistoryId != null) {
            syncSelectedPromptHistoryFromTextView()
        } else {
            addPromptHistoryIfNeeded(promptText, select = true)
        }
        val promptItem = selectedPromptHistoryId?.let { sid ->
            promptHistory.firstOrNull { it.id == sid }
        } ?: run {
            appendEventLine("请先语音输入或选择 Prompt")
            return
        }

        val ttsText = ttsTextInput.text.toString().trim()
        if (ttsText.isEmpty()) {
            appendEventLine("TTS 文本为空")
            return
        }
        if (selectedTTSTextHistoryId != null) {
            syncSelectedTTSTextHistoryFromTextView()
        } else {
            addTTSTextHistoryIfNeeded(ttsText, select = true)
        }

        startGeneration(promptItem = promptItem, promptText = promptText, ttsText = ttsText)
    }

    private fun startGeneration(promptItem: PromptHistoryItem, promptText: String, ttsText: String) {
        resetInferenceInfo("开始生成：PR${promptItem.index}")
        setOnlyRunningStage(InferenceStage.frontEnd)
        stopPlayback(pauseState = HistoryPlaybackState.Stopped)
        isGenerating = true
        generationSeq += 1
        val seq = generationSeq
        refreshControlsEnabled()

        generationJob?.cancel()
        generationJob = lifecycleScope.launch(Dispatchers.IO) {
            val resultPath = runCatching {
                val promptAudio = loadPromptMono16k(promptItem)
                inferenceBridge.runInference(
                    ttsText = ttsText,
                    promptText = promptText,
                    promptAudio = promptAudio,
                    promptSampleRate = inferencePromptSampleRate,
                    onEvent = { event -> handleInferenceEvent(event) },
                )
            }.getOrElse { t ->
                withContext(Dispatchers.Main) {
                    appendEventLine("failed before runInference: ${t.message}")
                }
                "false"
            }

            withContext(Dispatchers.Main) {
                if (seq != generationSeq) return@withContext
                if (resultPath == "false") {
                    appendEventLine("生成失败")
                } else {
                    addGeneratedHistory(resultPath, promptItem.index)
                    appendEventLine("生成完成")
                }
                isGenerating = false
                setOnlyRunningStage(null)
                renderInferenceInfo()
                refreshControlsEnabled()
            }
        }
    }

    private fun addGeneratedHistory(path: String, promptRefIndex: Int) {
        val file = File(path)
        if (!file.exists()) {
            appendEventLine("生成音频不存在: $path")
            return
        }
        nextGeneratedIndex += 1
        val duration = mediaDurationSeconds(file)
        val item = GeneratedHistoryItem(
            id = UUID.randomUUID().toString(),
            genIndex = nextGeneratedIndex,
            promptRefIndex = promptRefIndex,
            audioPath = file.absolutePath,
            durationSeconds = duration,
        )
        generatedHistory.add(0, item)
        if (generatedHistory.size > MAX_GENERATED_HISTORY) {
            val removed = generatedHistory.removeAt(generatedHistory.lastIndex)
            if (removed.audioPath != item.audioPath) {
                File(removed.audioPath).delete()
            }
        }
        selectedGeneratedHistoryId = item.id
        refreshGeneratedHistoryUI()
    }

    private fun historyPlayTapped() {
        val selected = selectedGeneratedHistoryId?.let { sid ->
            generatedHistory.firstOrNull { it.id == sid }
        }
        if (selected == null) {
            toast("请先选择一条生成历史")
            return
        }

        when (historyPlaybackState) {
            HistoryPlaybackState.Playing -> pauseHistoryPlayback()
            HistoryPlaybackState.Stopped, HistoryPlaybackState.Paused -> startPlayback(selected)
        }
    }

    private fun startPlayback(item: GeneratedHistoryItem) {
        stopPlayback(pauseState = HistoryPlaybackState.Stopped)
        runCatching {
            mediaPlayer = MediaPlayer().apply {
                setDataSource(item.audioPath)
                setOnCompletionListener {
                    stopPlayback(pauseState = HistoryPlaybackState.Stopped)
                }
                prepare()
                start()
            }
            historyPlaybackState = HistoryPlaybackState.Playing
            appendEventLine("开始播放 GR${item.genIndex}")
            refreshGeneratedHistoryUI()
            refreshControlsEnabled()
        }.onFailure { t ->
            appendEventLine("播放失败: ${t.message}")
            stopPlayback(pauseState = HistoryPlaybackState.Stopped)
        }
    }

    private fun pauseHistoryPlayback() {
        stopPlayback(pauseState = HistoryPlaybackState.Paused)
    }

    private fun stopPlayback(pauseState: HistoryPlaybackState) {
        mediaPlayer?.runCatching {
            stop()
            release()
        }
        mediaPlayer = null
        historyPlaybackState = pauseState
        refreshGeneratedHistoryUI()
        refreshControlsEnabled()
    }

    private fun handleInferenceEvent(event: InferenceEvent) {
        runOnUiThread {
            when (event) {
                is InferenceEvent.StageBegan -> {
                    setOnlyRunningStage(event.stage)
                    renderInferenceInfoThrottled(force = true)
                }
                is InferenceEvent.StageProgress -> {
                    stageLastUnitsDone[event.stage] = event.unitsDone
                    stageLastAvgUPS[event.stage] = event.avgUPS
                    setOnlyRunningStage(event.stage)
                    renderInferenceInfoThrottled(force = event.stage != InferenceStage.llm)
                }
                is InferenceEvent.StageEnded -> {
                    val stage = event.info.stage
                    stageEndedInfo[stage] = StageEndedDisplay(
                        seconds = event.info.seconds,
                        units = stageLastUnitsDone[stage] ?: event.info.units,
                        avgUPS = stageLastAvgUPS[stage] ?: event.info.avgUPS,
                    )
                    advanceRunningStage(after = stage)
                    appendEventLine("${stageDisplayName(stage)} ${"%.4f".format(event.info.seconds)}s")
                    renderInferenceInfoThrottled(force = true)
                }
                is InferenceEvent.Failed -> {
                    setOnlyRunningStage(null)
                    appendEventLine("${stageDisplayName(event.stage)} failed: ${event.message}")
                    renderInferenceInfoThrottled(force = true)
                }
                is InferenceEvent.AudioDuration -> {
                    audioDurationSeconds = event.seconds
                    appendEventLine("voiceGeneration audio ${"%.3f".format(event.seconds)}s")
                    renderInferenceInfoThrottled(force = true)
                }
            }
        }
    }

    private fun resetInferenceInfo(initialMessage: String) {
        eventLogLines.clear()
        stageEndedInfo.clear()
        stageRunning.clear()
        stageLastUnitsDone.clear()
        stageLastAvgUPS.clear()
        audioDurationSeconds = null
        lastProgressRenderMs = 0L
        appendEventLine(initialMessage)
        renderInferenceInfo()
    }

    private fun appendEventLine(line: String) {
        if (eventLogLines.size >= MAX_LOG_LINES) {
            eventLogLines.removeFirst()
        }
        eventLogLines.addLast(normalizeLogText(line))
        renderInferenceInfoThrottled(force = true)
    }

    private fun normalizeLogText(s: String): String =
        s.replace("hift", "HifiGan", ignoreCase = true)
            .replace("llm", "LLM", ignoreCase = true)
            .replace("flow", "Flow", ignoreCase = true)

    private fun setOnlyRunningStage(stage: InferenceStage?) {
        stageRunning.clear()
        if (stage != null) {
            stageRunning.add(stage)
        }
    }

    private fun advanceRunningStage(after: InferenceStage) {
        val idx = stageOrder.indexOf(after)
        if (idx == -1 || idx + 1 >= stageOrder.size) {
            setOnlyRunningStage(null)
            return
        }
        setOnlyRunningStage(stageOrder[idx + 1])
    }

    private fun renderInferenceInfoThrottled(force: Boolean) {
        val now = System.currentTimeMillis()
        if (force || now - lastProgressRenderMs >= 120) {
            lastProgressRenderMs = now
            renderInferenceInfo()
        }
    }

    private fun renderInferenceInfo() {
        val frontEndText = stageText(InferenceStage.frontEnd)
        val llmPrepareText = stageText(InferenceStage.llmPrepare)
        val llmText = llmStageText()
        val flowText = stageText(InferenceStage.flow)
        val hiftText = stageText(InferenceStage.hift)
        val voiceText = voiceStageText()

        val totalTime = listOf(
            stageEndedInfo[InferenceStage.llm]?.seconds,
            stageEndedInfo[InferenceStage.flow]?.seconds,
            stageEndedInfo[InferenceStage.hift]?.seconds,
        ).filterNotNull().takeIf { it.size == 3 }?.sum()

        val rtf = if (audioDurationSeconds != null && totalTime != null && audioDurationSeconds!! > 0) {
            totalTime / audioDurationSeconds!!
        } else {
            null
        }

        val sb = StringBuilder()
        sb.appendLine("Progress")
        sb.appendLine("────────────")
        sb.appendLine("[FrontEnd]        $frontEndText")
        sb.appendLine("[LLMPrepare]      $llmPrepareText")
        sb.appendLine("[LLM]             $llmText")
        sb.appendLine("[Flow]            $flowText")
        sb.appendLine("[HifiGan]         $hiftText")
        sb.appendLine("[voiceGeneration] $voiceText")
        sb.appendLine("[totalTime]       ${totalTime?.let { "%.4fs".format(it) } ?: "-"}")
        sb.appendLine("[RTF]             ${rtf?.let { "%.4f".format(it) } ?: "-"}")
        sb.appendLine()
        sb.appendLine("Events")
        sb.appendLine("────────────")
        eventLogLines.forEach { sb.appendLine(it) }
        inferenceInfoTv.text = sb.toString()
    }

    private fun stageText(stage: InferenceStage): String {
        val ended = stageEndedInfo[stage]
        if (ended != null) return "%.4fs".format(ended.seconds)
        if (stageRunning.contains(stage)) return "处理中..."
        return "-"
    }

    private fun llmStageText(): String {
        val ended = stageEndedInfo[InferenceStage.llm]
        if (ended != null) {
            val avg = if (ended.avgUPS > 0 && ended.avgUPS.isFinite()) {
                "%.2ftok/s".format(ended.avgUPS)
            } else {
                "-"
            }
            return "total=${ended.units}toks  time=${"%.4f".format(ended.seconds)}s  avg=$avg"
        }

        val total = stageLastUnitsDone[InferenceStage.llm]
        val avg = stageLastAvgUPS[InferenceStage.llm]
        if (total != null) {
            return if (avg != null && avg > 0 && avg.isFinite()) {
                "total=${total}toks  avg=${"%.2f".format(avg)}tok/s"
            } else {
                "total=${total}toks  avg=-"
            }
        }
        if (stageRunning.contains(InferenceStage.llm)) return "推理中..."
        return "-"
    }

    private fun voiceStageText(): String {
        val audio = audioDurationSeconds
        if (audio != null) return "audio=${"%.4f".format(audio)}s"
        val ended = stageEndedInfo[InferenceStage.voiceGeneration]
        if (ended != null) return "%.4fs".format(ended.seconds)
        if (stageRunning.contains(InferenceStage.voiceGeneration)) return "生成中..."
        return "-"
    }

    private fun stageDisplayName(stage: InferenceStage): String = when (stage) {
        InferenceStage.frontEnd -> "FrontEnd"
        InferenceStage.llmPrepare -> "LLMPrepare"
        InferenceStage.llm -> "LLM"
        InferenceStage.flow -> "Flow"
        InferenceStage.hift -> "HifiGan"
        InferenceStage.voiceGeneration -> "voiceGeneration"
    }

    private fun refreshControlsEnabled() {
        val currentState = captureState
        val hasMic = hasRecordPermission()
        val captureBusy = currentState != CaptureState.Idle
        val isPlaying = historyPlaybackState == HistoryPlaybackState.Playing

        generateButton.isEnabled = !isGenerating && !captureBusy && !isPlaying

        promptRecordButton.isEnabled = hasMic && !isGenerating && !isPlaying &&
            (currentState == CaptureState.Idle ||
                (currentState is CaptureState.Recording && currentState.target == CaptureTarget.Prompt))

        ttsRecordButton.isEnabled = hasMic && !isGenerating && !isPlaying &&
            (currentState == CaptureState.Idle ||
                (currentState is CaptureState.Recording && currentState.target == CaptureTarget.TtsText))

        promptHistoryButton.isEnabled = !isGenerating && !captureBusy && !isPlaying
        ttsHistoryButton.isEnabled = !isGenerating && !captureBusy && !isPlaying
        generatedHistoryButton.isEnabled = !isGenerating && !captureBusy && !isPlaying
        playHistoryButton.isEnabled = !isGenerating && !captureBusy

        promptTextInput.isEnabled =
            currentState == CaptureState.Idle && selectedPromptHistoryId != null && !isPlaying && !isGenerating
        ttsTextInput.isEnabled = !isGenerating && !captureBusy && !isPlaying

        when (currentState) {
            CaptureState.Idle -> {
                promptRecordButton.text = "语音输入"
                styleRecordPromptButton(active = true)
                ttsRecordButton.text = "语音输入"
                styleRecordTTSButton(active = true)
            }
            is CaptureState.Recording -> {
                if (currentState.target == CaptureTarget.Prompt) {
                    promptRecordButton.text = "停止"
                    styleRecordPromptButton(active = true)
                    ttsRecordButton.text = "语音输入"
                    styleDisabledButton(ttsRecordButton)
                } else {
                    promptRecordButton.text = "语音输入"
                    styleDisabledButton(promptRecordButton)
                    ttsRecordButton.text = "停止"
                    styleRecordTTSButton(active = true)
                }
            }
            is CaptureState.RecognizingFinal -> {
                if (currentState.target == CaptureTarget.Prompt) {
                    promptRecordButton.text = "识别中..."
                    styleDisabledButton(promptRecordButton)
                    ttsRecordButton.text = "语音输入"
                    styleDisabledButton(ttsRecordButton)
                } else {
                    promptRecordButton.text = "语音输入"
                    styleDisabledButton(promptRecordButton)
                    ttsRecordButton.text = "识别中..."
                    styleDisabledButton(ttsRecordButton)
                }
            }
        }

        styleNeutralButton(promptHistoryButton)
        styleNeutralButton(ttsHistoryButton)
        styleNeutralButton(generatedHistoryButton)
        if (generateButton.isEnabled) stylePrimaryButton(generateButton) else styleDisabledButton(generateButton)

        when (historyPlaybackState) {
            HistoryPlaybackState.Stopped -> {
                playHistoryButton.text = "播放"
                if (playHistoryButton.isEnabled) stylePlayButton() else styleDisabledButton(playHistoryButton)
            }
            HistoryPlaybackState.Playing -> {
                playHistoryButton.text = "暂停"
                if (playHistoryButton.isEnabled) stylePlayButton() else styleDisabledButton(playHistoryButton)
            }
            HistoryPlaybackState.Paused -> {
                playHistoryButton.text = "从头播放"
                if (playHistoryButton.isEnabled) stylePlayButton() else styleDisabledButton(playHistoryButton)
            }
        }

        if (promptHistoryButton.isEnabled) styleNeutralButton(promptHistoryButton) else styleDisabledButton(promptHistoryButton)
        if (ttsHistoryButton.isEnabled) styleNeutralButton(ttsHistoryButton) else styleDisabledButton(ttsHistoryButton)
        if (generatedHistoryButton.isEnabled) styleNeutralButton(generatedHistoryButton) else styleDisabledButton(generatedHistoryButton)
    }

    private fun refreshPromptHistoryUI() {
        val selected = selectedPromptHistoryId?.let { sid -> promptHistory.firstOrNull { it.id == sid } }
        promptHistoryButton.text = if (selected == null) {
            if (promptHistory.isEmpty()) "Prompt: Empty" else "Prompt: Select"
        } else {
            "Prompt: ${promptHistoryLabel(selected)}"
        }
    }

    private fun refreshTTSHistoryUI() {
        val selected = selectedTTSTextHistoryId?.let { sid -> ttsTextHistory.firstOrNull { it.id == sid } }
        ttsHistoryButton.text = if (selected == null) {
            if (ttsTextHistory.isEmpty()) "TTS历史：暂无" else "TTS历史：请选择"
        } else {
            "TTS历史：${ttsTextHistoryLabel(selected)}"
        }
    }

    private fun refreshGeneratedHistoryUI() {
        val selected = selectedGeneratedHistoryId?.let { sid -> generatedHistory.firstOrNull { it.id == sid } }
        generatedHistoryButton.text = if (selected == null) {
            if (generatedHistory.isEmpty()) "合成历史：暂无" else "请选择合成历史"
        } else {
            "合成历史：${generatedHistoryLabel(selected)}"
        }
    }

    private fun syncSelectedPromptHistoryFromTextView() {
        val text = promptTextInput.text?.toString() ?: ""
        val sid = selectedPromptHistoryId ?: return
        val idx = promptHistory.indexOfFirst { it.id == sid }
        if (idx >= 0) {
            promptHistory[idx] = promptHistory[idx].copy(text = text)
            refreshPromptHistoryUI()
        }
    }

    private fun syncSelectedTTSTextHistoryFromTextView() {
        val text = ttsTextInput.text?.toString() ?: ""
        val sid = selectedTTSTextHistoryId ?: return
        val idx = ttsTextHistory.indexOfFirst { it.id == sid }
        if (idx >= 0) {
            ttsTextHistory[idx] = ttsTextHistory[idx].copy(text = text)
            refreshTTSHistoryUI()
        }
    }

    private fun applyRecognizedText(target: CaptureTarget, text: String) {
        when (target) {
            CaptureTarget.Prompt -> promptTextInput.setText(text)
            CaptureTarget.TtsText -> ttsTextInput.setText(text)
        }
    }

    private fun resultsToText(results: Bundle?): String? =
        results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.firstOrNull()

    private fun styleRecordPromptButton(active: Boolean) {
        val bg = if (active) COLOR_SYSTEM_GREEN else COLOR_SYSTEM_GRAY
        applyButtonStyle(promptRecordButton, bg, Color.WHITE)
    }

    private fun styleRecordTTSButton(active: Boolean) {
        val bg = if (active) COLOR_SYSTEM_GREEN else COLOR_SYSTEM_GRAY
        applyButtonStyle(ttsRecordButton, bg, Color.WHITE)
    }

    private fun stylePrimaryButton(button: Button) {
        applyButtonStyle(button, COLOR_SYSTEM_BLUE, Color.WHITE)
    }

    private fun stylePlayButton() {
        applyButtonStyle(playHistoryButton, COLOR_SYSTEM_ORANGE, Color.WHITE)
    }

    private fun styleNeutralButton(button: Button) {
        applyButtonStyle(button, COLOR_SYSTEM_GRAY6, COLOR_SYSTEM_BLUE)
    }

    private fun styleDisabledButton(button: Button) {
        applyButtonStyle(button, COLOR_SYSTEM_GRAY, Color.WHITE)
    }

    private fun applyButtonStyle(button: Button, bgColor: Int, textColor: Int) {
        button.backgroundTintList = null
        button.background = roundedRect(bgColor)
        button.setTextColor(textColor)
        button.alpha = if (button.isEnabled) 1.0f else 0.72f
    }

    private fun roundedRect(color: Int): GradientDrawable =
        GradientDrawable().apply {
            shape = GradientDrawable.RECTANGLE
            cornerRadius = dp(CORNER_RADIUS_DP)
            setColor(color)
        }

    private fun dp(value: Float): Float =
        TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, value, resources.displayMetrics)

    private fun mediaDurationSeconds(file: File): Double {
        val retriever = MediaMetadataRetriever()
        return try {
            retriever.setDataSource(file.absolutePath)
            val ms = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                ?.toLongOrNull() ?: 0L
            ms / 1000.0
        } finally {
            retriever.release()
        }
    }

    private fun installPresetPromptAudio(index: Int): String? {
        val candidates = PRESET_PROMPT_AUDIO_ASSETS.getOrNull(index - 1) ?: return null
        val outDir = File(filesDir, DIRECTORY_PROMPT_AUDIO).apply { mkdirs() }
        val outFile = File(outDir, "prompt_$index.wav")
        for (assetPath in candidates) {
            if (copyAssetToFileIfExists(assetPath, outFile)) {
                return outFile.absolutePath
            }
        }
        return null
    }

    private fun copyAssetToFileIfExists(assetPath: String, outFile: File): Boolean {
        if (outFile.exists() && outFile.length() > 0L) {
            return true
        }
        return try {
            assets.open(assetPath).use { input ->
                outFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
            true
        } catch (_: IOException) {
            false
        }
    }

    private fun loadPromptMono16k(promptItem: PromptHistoryItem): FloatArray {
        val path = resolvePromptAudioPath(promptItem)
            ?: error("Prompt #${promptItem.index} 没有音频文件，请确认 assets/local_llm_resources/prompt_${promptItem.index}.wav 存在")
        val file = File(path)
        if (!file.exists()) {
            error("Prompt 音频文件不存在: $path")
        }

        val wav = readWavAudio(file)
        val mono = if (wav.channels == 1) {
            wav.samples
        } else {
            val frameCount = wav.samples.size / wav.channels
            FloatArray(frameCount) { frame ->
                var sum = 0f
                var c = 0
                while (c < wav.channels) {
                    sum += wav.samples[frame * wav.channels + c]
                    c += 1
                }
                sum / wav.channels.toFloat()
            }
        }

        val srcRate = wav.sampleRate
        return if (srcRate == inferencePromptSampleRate) {
            mono
        } else {
            linearResample(mono, srcRate, inferencePromptSampleRate)
        }
    }

    private fun linearResample(input: FloatArray, srcRate: Int, dstRate: Int): FloatArray {
        if (input.isEmpty() || srcRate <= 0 || dstRate <= 0 || srcRate == dstRate) return input
        val outLength = ((input.size.toDouble() * dstRate.toDouble()) / srcRate.toDouble())
            .toInt()
            .coerceAtLeast(1)
        val output = FloatArray(outLength)
        val ratio = srcRate.toDouble() / dstRate.toDouble()
        var i = 0
        while (i < outLength) {
            val srcPos = i * ratio
            val left = srcPos.toInt().coerceIn(0, input.lastIndex)
            val right = (left + 1).coerceAtMost(input.lastIndex)
            val frac = (srcPos - left).toFloat()
            output[i] = input[left] * (1f - frac) + input[right] * frac
            i += 1
        }
        return output
    }

    private fun readWavAudio(file: File): WavAudio {
        val bytes = file.readBytes()
        require(bytes.size >= 44) { "WAV 文件太小: ${file.absolutePath}" }

        val riff = String(bytes, 0, 4, Charsets.US_ASCII)
        val wave = String(bytes, 8, 4, Charsets.US_ASCII)
        require(riff == "RIFF" && wave == "WAVE") { "不是标准 WAV 文件: ${file.absolutePath}" }

        var offset = 12
        var channels = 0
        var sampleRate = 0
        var bitsPerSample = 0
        var audioFormat = 0
        var dataOffset = -1
        var dataSize = 0

        while (offset + 8 <= bytes.size) {
            val chunkId = String(bytes, offset, 4, Charsets.US_ASCII)
            val chunkSize = ByteBuffer.wrap(bytes, offset + 4, 4)
                .order(ByteOrder.LITTLE_ENDIAN)
                .int
            val payloadStart = offset + 8
            val payloadEnd = payloadStart + chunkSize
            if (payloadEnd > bytes.size) break

            when (chunkId) {
                "fmt " -> {
                    require(chunkSize >= 16) { "WAV fmt chunk 无效: ${file.absolutePath}" }
                    val fmt = ByteBuffer.wrap(bytes, payloadStart, chunkSize)
                        .order(ByteOrder.LITTLE_ENDIAN)
                    audioFormat = fmt.short.toInt() and 0xFFFF
                    channels = fmt.short.toInt() and 0xFFFF
                    sampleRate = fmt.int
                    fmt.int // byteRate
                    fmt.short // blockAlign
                    bitsPerSample = fmt.short.toInt() and 0xFFFF
                }
                "data" -> {
                    dataOffset = payloadStart
                    dataSize = chunkSize
                }
            }
            offset = payloadEnd + (chunkSize and 1)
        }

        require(channels > 0 && sampleRate > 0 && dataOffset >= 0 && dataSize > 0) {
            "WAV 缺少必要信息: ${file.absolutePath}"
        }
        val bytesPerSample = bitsPerSample / 8
        require(bytesPerSample > 0) { "不支持的位深: $bitsPerSample" }
        val frameStride = bytesPerSample * channels
        require(frameStride > 0) { "WAV 声道/位深非法: ${file.absolutePath}" }
        val frameCount = dataSize / frameStride
        require(frameCount > 0) { "WAV 数据为空: ${file.absolutePath}" }

        val sampleCount = frameCount * channels
        val samples = FloatArray(sampleCount)
        val buffer = ByteBuffer.wrap(bytes, dataOffset, frameCount * frameStride)
            .order(ByteOrder.LITTLE_ENDIAN)

        var i = 0
        while (i < sampleCount) {
            samples[i] = when {
                audioFormat == 1 && bitsPerSample == 16 -> {
                    (buffer.short.toInt() / 32768.0f).coerceIn(-1.0f, 1.0f)
                }
                audioFormat == 1 && bitsPerSample == 32 -> {
                    (buffer.int / 2147483648.0f).coerceIn(-1.0f, 1.0f)
                }
                audioFormat == 3 && bitsPerSample == 32 -> {
                    buffer.float.coerceIn(-1.0f, 1.0f)
                }
                else -> error("暂不支持的 WAV 格式: format=$audioFormat bits=$bitsPerSample")
            }
            i += 1
        }
        return WavAudio(samples = samples, sampleRate = sampleRate, channels = channels)
    }

    private fun resolvePromptAudioPath(promptItem: PromptHistoryItem): String? {
        val current = promptItem.audioPath
        if (current != null && File(current).exists()) {
            return current
        }

        val indexedLocal = File(File(filesDir, DIRECTORY_PROMPT_AUDIO), "prompt_${promptItem.index}.wav")
        if (indexedLocal.exists()) {
            bindPromptAudioPath(promptItem.id, indexedLocal.absolutePath)
            return indexedLocal.absolutePath
        }

        val installed = installPresetPromptAudio(index = promptItem.index)
        if (installed != null && File(installed).exists()) {
            bindPromptAudioPath(promptItem.id, installed)
            return installed
        }

        val fallback = promptHistory.firstNotNullOfOrNull { item ->
            item.audioPath?.takeIf { File(it).exists() }
        }
        if (fallback != null) {
            bindPromptAudioPath(promptItem.id, fallback)
            return fallback
        }
        return null
    }

    private fun bindPromptAudioPath(promptId: String, audioPath: String) {
        val idx = promptHistory.indexOfFirst { it.id == promptId }
        if (idx >= 0 && promptHistory[idx].audioPath != audioPath) {
            promptHistory[idx] = promptHistory[idx].copy(audioPath = audioPath)
        }
    }

    private fun promptHistoryLabel(item: PromptHistoryItem): String =
        "PR${item.index} | ${item.text.trim().length} chars"

    private fun ttsTextHistoryLabel(item: TTSTextHistoryItem): String {
        val trimmed = item.text.trim()
        return "T${item.index} | ${trimmed.length} chars"
    }

    private fun generatedHistoryLabel(item: GeneratedHistoryItem): String =
        "#${item.genIndex} | PR${item.promptRefIndex} | ${"%.2fs".format(item.durationSeconds)}"

    private fun hasRecordPermission(): Boolean =
        ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.RECORD_AUDIO,
        ) == android.content.pm.PackageManager.PERMISSION_GRANTED

    private fun toast(msg: String) {
        Toast.makeText(this, msg, Toast.LENGTH_SHORT).show()
    }

    private fun hideKeyboardAndClearFocus(view: android.view.View?) {
        val target = view ?: return
        val imm = getSystemService(INPUT_METHOD_SERVICE) as? InputMethodManager
        imm?.hideSoftInputFromWindow(target.windowToken, 0)
        target.clearFocus()
        findViewById<ScrollView>(R.id.root_scroll).requestFocus()
    }

    override fun onStop() {
        generationJob?.cancel()
        if (captureState is CaptureState.Recording) {
            stopCurrentCaptureAndWaitFinal()
        }
        super.onStop()
    }

    override fun onDestroy() {
        generationJob?.cancel()
        stopPlayback(pauseState = HistoryPlaybackState.Stopped)
        speechRecognizer?.destroy()
        speechRecognizer = null
        inferenceBridge.close()
        super.onDestroy()
    }

    companion object {
        private const val MAX_PROMPT_HISTORY = 30
        private const val MAX_TTS_HISTORY = 30
        private const val MAX_GENERATED_HISTORY = 30
        private const val MAX_LOG_LINES = 20

        private val PRESET_PROMPT_TEXTS = listOf(
            "对，这就是我，万人敬仰的太乙真人，虽然有点婴儿肥，但也掩不住我逼人的帅气。",
            "今夜的月光如此清亮，不做些什么真是浪费。随我一同去月下漫步吧，不许拒绝。",
        )

        private val COLOR_SYSTEM_GREEN = Color.parseColor("#03fc41")
        private val COLOR_SYSTEM_BLUE = Color.parseColor("#007AFF")
        private val COLOR_SYSTEM_ORANGE = Color.parseColor("#FF9500")
        private val COLOR_SYSTEM_GRAY = Color.parseColor("#8E8E93")
        private val COLOR_SYSTEM_GRAY5 = Color.parseColor("#E5E5EA")
        private val COLOR_SYSTEM_GRAY6 = Color.parseColor("#F2F2F7")
        private val COLOR_LABEL = Color.parseColor("#1C1C1E")
        private const val CORNER_RADIUS_DP = 8f
        private const val DIRECTORY_PROMPT_AUDIO = "prompt_audio"
        private val PRESET_PROMPT_AUDIO_ASSETS = listOf(
            listOf("local_llm_resources/prompt_1.wav", "prompts/prompt_1.wav", "prompt_1.wav"),
            listOf("local_llm_resources/prompt_2.wav", "prompts/prompt_2.wav", "prompt_2.wav"),
        )
    }
}

private enum class CaptureTarget {
    Prompt,
    TtsText,
}

private sealed class CaptureState {
    data object Idle : CaptureState()
    data class Recording(val target: CaptureTarget, val sessionId: String) : CaptureState()
    data class RecognizingFinal(val target: CaptureTarget, val sessionId: String) : CaptureState()
}

private enum class HistoryPlaybackState {
    Stopped,
    Playing,
    Paused,
}

private data class PromptHistoryItem(
    val id: String,
    val index: Int,
    val text: String,
    val audioPath: String?,
)

private data class WavAudio(
    val samples: FloatArray,
    val sampleRate: Int,
    val channels: Int,
)

private data class TTSTextHistoryItem(
    val id: String,
    val index: Int,
    val text: String,
)

private data class GeneratedHistoryItem(
    val id: String,
    val genIndex: Int,
    val promptRefIndex: Int,
    val audioPath: String,
    val durationSeconds: Double,
)

private data class StageEndedDisplay(
    val seconds: Double,
    val units: Int,
    val avgUPS: Double,
)
