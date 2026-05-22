# Project-specific ProGuard rules.

-keep class com.example.llama.QwenTokenizerService { *; }
-keep class ai.onnxruntime.** { *; }
-keep class com.google.ai.edge.litert.** { *; }

# Keep JNI bridge classes/constructors stable for native NewObject/FindClass.
-keep class com.example.llama.LiteRtNativeFlowRunner { *; }
-keep class com.example.llama.NativeCreateResult { *; }
