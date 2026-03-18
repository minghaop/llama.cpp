# Android 上集成 llama.cpp 进行 TTS 推理

## 概述

这个文档说明了如何在 Android TTS 应用中集成 llama.cpp 进行语音合成推理。

## 项目结构

```
app/
├── src/main/
│   ├── java/com/example/localllmclient/
│   │   ├── MainActivity.kt
│   │   └── NativeLib.kt          # JNI 封装
│   ├── cpp/
│   │   ├── CMakeLists.txt        # CMake 配置
│   │   └── localllmclient.cpp    # JNI 绑定
│   └── jniLibs/                  # 原生库
│       └── arm64-v8a/
│           └── liblocalllmclient.so
```

## 步骤 1：配置 build.gradle.kts

在 `app/build.gradle.kts` 中配置：

```kotlin
android {
    defaultConfig {
        externalNativeBuild {
            cmake {
                arguments += "-DANDROID_STL=c++_shared"
            }
        }
        ndk {
            abiFilters += listOf("arm64-v8a", "armeabi-v7a")
        }
    }
    
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
        }
    }
}
```

## 步骤 2：NativeLib.kt

JNI 接口定义：

```kotlin
object NativeLib {
    init {
        System.loadLibrary("localllmclient")
    }

    external fun initBackend()
    external fun loadModel(modelPath: String): Int
    external fun initContext(nCtx: Int, nBatch: Int, nThreads: Int): Int
    external fun runInference(
        ttsText: String,
        promptText: String,
        promptAudioPath: String,
        promptSampleRate: Int,
        promptAudioLen: Int
    ): String
    external fun cleanup()
    external fun getSystemInfo(): String
    external fun isModelLoaded(): Boolean
}
```

## 步骤 3：C++ 实现 (localllmclient.cpp)

主要功能：

1. **llama.cpp 后端初始化**
2. **模型加载**
3. **上下文初始化**
4. **语音合成推理** (`runInference`)

### runInference 函数流程

1. 加载 Prompt 音频文件 (WAV, float32)
2. 重采样到 16kHz
3. 调用 CosyVoice 推理逻辑
4. 保存生成的音频文件

```cpp
JNIEXPORT jstring JNICALL
Java_com_example_localllmclient_NativeLib_runInference(
    JNIEnv *env, jobject /* this */,
    jstring jtts_text,
    jstring jprompt_text,
    jstring jprompt_audio_path,
    jint prompt_sample_rate,
    jint prompt_audio_len) {
    // 实现细节...
}
```

## 步骤 4：MainActivity.kt

### 初始化 llama.cpp

```kotlin
private fun initLlamaBackend() {
    Thread {
        try {
            NativeLib.initBackend()
            
            val modelPath = getExternalFilesDir(null)?.absolutePath + "/models/cosyvoice2-0.5B-Q2_K.gguf"
            val loadResult = NativeLib.loadModel(modelPath)
            
            if (loadResult == 0) {
                NativeLib.initContext(2048, 512, 4)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error initializing llama.cpp", e)
        }
    }.start()
}
```

### 开始生成

```kotlin
private fun startGeneration(promptText: String, ttsText: String) {
    val result = NativeLib.runInference(
        ttsText = ttsText,
        promptText = promptText,
        promptAudioPath = promptItem.audioPath,
        promptSampleRate = 16000,
        promptAudioLen = 0
    )
    // 处理结果...
}
```

## 步骤 5：编译原生库

### 使用 Android Studio

1. 点击 "Build" -> "Build Bundle(s) / APK(s)" -> "Build APK(s)"
2. Android Studio 会自动编译原生库

### 使用命令行

```bash
cd Example/AndroidTTSApp
./gradlew assembleDebug
```

## 步骤 6：放置模型文件

将 `.gguf` 格式的模型文件放到设备的以下目录：

```
Android/data/com.example.localllmclient/files/models/cosyvoice2-0.5B-Q2_K.gguf
```

或者在应用首次启动时下载模型。

## llama.cpp 主要 API

| API | 描述 |
|-----|------|
| `llama_backend_init` | 初始化 llama.cpp 后端 |
| `llama_model_load_from_file` | 从文件加载模型 |
| `llama_init_from_model` | 从模型创建上下文 |
| `llama_tokenize` | 文本分词 |
| `llama_decode` | 解码生成 |
| `llama_kv_cache_clear` | 清除 KV 缓存 |
| `llama_free` | 释放上下文 |
| `llama_model_free` | 释放模型 |

## 注意事项

1. **模型格式**：Android 上只能使用 `.gguf` 格式的模型
2. **内存限制**：Android 设备内存有限，建议使用量化后的模型（Q2_K, Q3_K_S 等）
3. **线程安全**：确保在正确的线程上调用 JNI 方法
4. **资源释放**：在 `onDestroy` 中调用 `cleanup()` 释放资源
5. **音频格式**：Prompt 音频应该是单声道、float32 格式的 WAV 文件

## 当前实现状态

- ✅ llama.cpp 后端初始化
- ✅ 模型加载
- ✅ 上下文初始化
- ✅ 文本生成
- ⏳ 语音合成推理 (需要集成 CosyVoice C++ 代码)
- ✅ WAV 文件读写
- ✅ 音频重采样
- ✅ UI 集成

## 下一步

要完成完整的 TTS 功能，需要：

1. 集成 CosyVoice 的 C++ 推理代码
2. 实现音频特征提取 (Mel spectrogram)
3. 实现 Flow 和 HIFT 推理
4. 实现 ISTFT (逆短时傅里叶变换)

## 参考实现

- llama.cpp/examples/llama.android
- Sources/LocalLLMClientLlama/Context.swift