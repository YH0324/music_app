package com.example.music_app

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.nio.FloatBuffer

/**
 * 只負責「跑 yolo.onnx 拿 raw 輸出」。
 * letterbox / decode / NMS 都在 Python(yolo_onnx.py)。
 */
class YoloOrt(modelPath: String) {
    private val env = OrtEnvironment.getEnvironment()
    private val session = env.createSession(modelPath, OrtSession.SessionOptions())
    private val inputName = session.inputNames.first()

    /**
     * data: letterbox 後的影像,已攤平成 1 維,長度 = 1*3*h*w(CHW 順序、值已正規化到 0~1)
     * 回傳: Pair(攤平的 raw 輸出, 形狀 IntArray)  例如 ([...], [1,26,21504])
     */
    fun run(data: FloatArray, h: Int, w: Int): Pair<FloatArray, IntArray> {
        val shape = longArrayOf(1, 3, h.toLong(), w.toLong())
        val tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(data), shape)
        tensor.use {
            session.run(mapOf(inputName to it)).use { result ->
                val out = result[0] as OnnxTensor
                val info = out.info as ai.onnxruntime.TensorInfo
                val outShape = info.shape.map { d -> d.toInt() }.toIntArray()
                val flat = out.floatBuffer.let { fb ->
                    FloatArray(fb.remaining()).also { arr -> fb.get(arr) }
                }
                return Pair(flat, outShape)
            }
        }
    }

    companion object {
        /** 從 assets 複製 yolo.onnx 到 filesDir 並建立 YoloOrt。 */
        fun fromAssets(ctx: android.content.Context): YoloOrt {
            val f = File(ctx.filesDir, "yolo.onnx")
            if (!f.exists()) {
                ctx.assets.open("yolo.onnx").use { input ->
                    f.outputStream().use { input.copyTo(it) }
                }
            }
            return YoloOrt(f.absolutePath)
        }
    }
}