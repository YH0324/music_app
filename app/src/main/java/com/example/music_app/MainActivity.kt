package com.example.music_app

import android.net.Uri
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import java.io.File

class MainActivity : AppCompatActivity() {

    private var midiPath: String? = null
    private lateinit var tv: TextView
    private lateinit var btnPlay: Button

    // 相簿選圖
    private val pickImage = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        if (uri != null) recognize(uri)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(this))
        }

        tv = findViewById(R.id.textView)
        btnPlay = findViewById(R.id.btnPlay)

        findViewById<Button>(R.id.btnPick).setOnClickListener {
            pickImage.launch("image/*")
        }
        btnPlay.setOnClickListener { playMidi() }
    }

    private fun recognize(uri: Uri) {
        tv.text = "辨識中...(第一次較久)"
        btnPlay.isEnabled = false
        Thread {
            val result = try {
                // 把選到的圖複製成 filesDir/input.png
                val imgFile = File(filesDir, "input.png")
                contentResolver.openInputStream(uri)!!.use { input ->
                    imgFile.outputStream().use { input.copyTo(it) }
                }
                val yolo = YoloOrt.fromAssets(this)
                val py = Python.getInstance()
                val r = py.getModule("entry").callAttr(
                    "recognize_image",
                    imgFile.absolutePath,
                    filesDir.absolutePath,
                    filesDir.absolutePath,
                    yolo
                ).toString()
                // MIDI 固定在 filesDir/input_output.mid
                midiPath = File(filesDir, "input_output.mid").absolutePath
                r
            } catch (e: Exception) {
                "錯誤: ${e.message}\n${e.stackTraceToString().take(500)}"
            }
            runOnUiThread {
                tv.text = result
                btnPlay.isEnabled = (midiPath != null && File(midiPath!!).exists())
            }
        }.start()
    }

    private fun playMidi() {
        val path = midiPath ?: return
        try {
            val mp = android.media.MediaPlayer()
            mp.setDataSource(path)
            mp.setOnPreparedListener { it.start() }
            mp.setOnCompletionListener { it.release() }
            mp.prepareAsync()
            tv.text = "播放中... 🎵"
        } catch (e: Exception) {
            tv.text = "播放錯誤: ${e.message}"
        }
    }
}