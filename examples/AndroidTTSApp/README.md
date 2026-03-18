# LocalLLMClient Android TTS Application

这是一个 Android 版本的文本转语音（TTS）应用程序，基于 LocalLLMClient 实现。

## 功能

- **Prompt 录制**：支持录制 Prompt 音频并自动识别成文字（ASR）
- **TTS 文本输入**：支持手动输入、语音输入和历史记录选择
- **语音生成**：调用 LLM 推理生成语音
- **历史播放**：播放生成的语音历史记录
- **推理进度显示**：显示各阶段的推理进度和日志

## 项目结构

```
AndroidTTSApp/
├── app/
│   ├── src/
│   │   └── main/
│   │       ├── java/com/example/localllmclient/
│   │       │   └── MainActivity.kt
│   │       ├── res/
│   │       │   ├── layout/
│   │       │   │   └── activity_main.xml
│   │       │   ├── values/
│   │       │   │   ├── strings.xml
│   │       │   │   └── themes.xml
│   │       │   ├── xml/
│   │       │   │   ├── data_extraction_rules.xml
│   │       │   │   └── backup_rules.xml
│   │       │   └── mipmap/
│   │       │       ├── ic_launcher.xml
│   │       │       └── ic_launcher_round.xml
│   │       └── AndroidManifest.xml
│   └── build.gradle.kts
├── settings.gradle.kts
└── gradle.properties
```

## 依赖

- AndroidX Core KTX
- AppCompat
- Material Components

## 权限

- RECORD_AUDIO：用于录制音频
- INTERNET：用于网络访问
- WRITE_EXTERNAL_STORAGE：用于保存生成的音频
- READ_EXTERNAL_STORAGE：用于读取音频文件
- MODIFY_AUDIO_SETTINGS：用于修改音频设置
- FOREGROUND_SERVICE：用于前台服务

## 使用说明

1. 克隆项目到本地
2. 使用 Android Studio 打开项目
3. 连接 Android 设备或启动模拟器
4. 点击 Run 按钮运行应用

## 注意事项

1. 首次运行需要授予录音权限
2. 预设的音频文件需要手动添加到 `presets/` 目录
3. 实际的 LLM 推理需要集成 LocalLLMClient 的 Android 版本

## 与 iOS 版本的差异

由于平台差异，某些功能可能需要调整：

- iOS 使用 AVAudioEngine 进行录音，Android 使用 AudioRecord
- iOS 使用 SFSpeechRecognizer 进行语音识别，Android 需要使用 Google Speech-to-Text API 或其他 ASR 服务
- iOS 使用 AudioPlayerMemory 播放音频，Android 使用 AudioTrack