package com.edgeai.signedge

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Matrix
import android.os.Build
import android.os.Bundle
import android.os.Debug
import android.os.SystemClock
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.util.Log
import android.util.Size
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.io.File
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.min

class MainActivity : AppCompatActivity() {

    companion object {
        /** Side of the centre ROI as a fraction of the shorter image side. */
        const val ROI_FRACTION = 0.7f
        /** Actuation rule: confidence >= threshold for HOLD_FRAMES consecutive frames. */
        const val CONF_THRESHOLD = 0.80f
        const val HOLD_FRAMES = 5
        const val NOTHING = "nothing"
        private const val LOG_EVERY = 100
    }

    private lateinit var previewView: PreviewView
    private lateinit var overlay: RoiOverlayView
    private lateinit var modelInputView: ImageView
    private lateinit var statsView: TextView
    private lateinit var predictionView: TextView
    private lateinit var wordView: TextView

    private lateinit var classifier: SignClassifier
    private val stats = LatencyStats()
    private val analysisExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private var camera: Camera? = null
    private var lensFacing = CameraSelector.LENS_FACING_BACK

    // Touched only on the analysis thread.
    private var stableLabel = ""
    private var stableCount = 0

    private val word = StringBuilder()
    private var apkSizeMb = 0.0

    private val permissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) startCamera()
            else Toast.makeText(this, "Camera permission is required", Toast.LENGTH_LONG).show()
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        previewView = findViewById(R.id.preview)
        overlay = findViewById(R.id.overlay)
        modelInputView = findViewById(R.id.modelInput)
        statsView = findViewById(R.id.stats)
        predictionView = findViewById(R.id.prediction)
        wordView = findViewById(R.id.word)
        previewView.scaleType = PreviewView.ScaleType.FIT_CENTER

        // setup(): load model from "flash" and allocate tensors.
        classifier = SignClassifier(this)
        apkSizeMb = File(applicationInfo.sourceDir).length() / (1024.0 * 1024.0)

        findViewById<Button>(R.id.btnSwitch).setOnClickListener {
            lensFacing = if (lensFacing == CameraSelector.LENS_FACING_BACK)
                CameraSelector.LENS_FACING_FRONT else CameraSelector.LENS_FACING_BACK
            startCamera()
        }
        findViewById<Button>(R.id.btnBackspace).setOnClickListener {
            if (word.isNotEmpty()) word.setLength(word.length - 1)
            wordView.text = word
        }
        findViewById<Button>(R.id.btnClear).setOnClickListener {
            word.clear(); wordView.text = word
        }
        findViewById<Button>(R.id.btnReset).setOnClickListener { stats.reset() }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED
        ) startCamera() else permissionLauncher.launch(Manifest.permission.CAMERA)
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            val aspect = AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY

            val preview = Preview.Builder()
                .setResolutionSelector(ResolutionSelector.Builder().setAspectRatioStrategy(aspect).build())
                .build()
                .also { it.setSurfaceProvider(previewView.surfaceProvider) }

            // The "sensor": small 640x480 RGBA frames, always the newest one (drop stale frames).
            val analysis = ImageAnalysis.Builder()
                .setResolutionSelector(
                    ResolutionSelector.Builder()
                        .setAspectRatioStrategy(aspect)
                        .setResolutionStrategy(
                            ResolutionStrategy(
                                Size(640, 480),
                                ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
                            )
                        ).build()
                )
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also { it.setAnalyzer(analysisExecutor, ::analyze) }

            var selector = CameraSelector.Builder().requireLensFacing(lensFacing).build()
            if (!provider.hasCamera(selector)) selector = CameraSelector.DEFAULT_BACK_CAMERA
            provider.unbindAll()
            camera = provider.bindToLifecycle(this, selector, preview, analysis)
        }, ContextCompat.getMainExecutor(this))
    }

    /** loop(): one frame -> preprocess -> Invoke -> threshold -> actuate. */
    private fun analyze(image: ImageProxy) {
        val t0 = SystemClock.elapsedRealtimeNanos()
        val rotation = image.imageInfo.rotationDegrees
        val (uprightW, uprightH) =
            if (rotation % 180 == 0) image.width to image.height else image.height to image.width
        val roi = image.use { cropRoi(it.toBitmap(), rotation) }

        val pred = classifier.classify(roi)
        val totalMs = (SystemClock.elapsedRealtimeNanos() - t0) / 1e6
        stats.add(pred.inferenceMs, totalMs)

        if (pred.confidence >= CONF_THRESHOLD && pred.label == stableLabel) {
            stableCount++
        } else if (pred.confidence >= CONF_THRESHOLD) {
            stableLabel = pred.label; stableCount = 1
        } else {
            stableLabel = ""; stableCount = 0
        }
        // Fires exactly once per held sign; repeat a letter by dropping the hand briefly.
        val fire = stableCount == HOLD_FRAMES && stableLabel != NOTHING

        if (stats.frames % LOG_EVERY == 0L) logSummary()

        runOnUiThread {
            overlay.setImageSize(uprightW, uprightH)
            modelInputView.setImageBitmap(roi)
            predictionView.text = "%s  %.0f%%".format(pred.label, pred.confidence * 100)
            predictionView.setTextColor(
                if (pred.confidence >= CONF_THRESHOLD) Color.WHITE else Color.GRAY
            )
            statsView.text = statsText(pred.inferenceMs, totalMs)
            if (fire) actuate(pred)
        }
    }

    /** Centre square crop -> rotate upright -> area-like downscale to the model input size. */
    private fun cropRoi(frame: Bitmap, rotation: Int): Bitmap {
        val side = (min(frame.width, frame.height) * ROI_FRACTION).toInt()
        val left = (frame.width - side) / 2
        val top = (frame.height - side) / 2
        val m = Matrix().apply { postRotate(rotation.toFloat()) }
        var bmp = Bitmap.createBitmap(frame, left, top, side, side, m, true)
        // Halve repeatedly before the final resize to avoid aliasing (close to cv2.INTER_AREA).
        val target = classifier.inputSize
        while (bmp.width / 2 >= target * 2) {
            bmp = Bitmap.createScaledBitmap(bmp, bmp.width / 2, bmp.height / 2, true)
        }
        return Bitmap.createScaledBitmap(bmp, target, target, true)
    }

    /** Physical actuation: torch pulse + vibration + UI highlight. */
    private fun actuate(pred: Prediction) {
        Log.i(TAG, "ACTUATE letter=${pred.label} conf=%.3f".format(pred.confidence))
        word.append(pred.label)
        wordView.text = word

        vibrator()?.let { v ->
            if (Build.VERSION.SDK_INT >= 26) {
                v.vibrate(VibrationEffect.createOneShot(120, VibrationEffect.DEFAULT_AMPLITUDE))
            } else {
                @Suppress("DEPRECATION") v.vibrate(120)
            }
        }
        camera?.takeIf { it.cameraInfo.hasFlashUnit() }?.let { cam ->
            cam.cameraControl.enableTorch(true)
            predictionView.postDelayed({ cam.cameraControl.enableTorch(false) }, 300)
        }
        predictionView.setTextColor(Color.GREEN)
        overlay.setAccepted(true)
        predictionView.postDelayed({ overlay.setAccepted(false) }, 400)
    }

    private fun vibrator(): Vibrator? =
        if (Build.VERSION.SDK_INT >= 31) {
            (getSystemService(VIBRATOR_MANAGER_SERVICE) as? VibratorManager)?.defaultVibrator
        } else {
            @Suppress("DEPRECATION") getSystemService(VIBRATOR_SERVICE) as? Vibrator
        }

    private fun statsText(inferMs: Double, totalMs: Double): String {
        val rt = Runtime.getRuntime()
        val javaMb = (rt.totalMemory() - rt.freeMemory()) / (1024.0 * 1024.0)
        val nativeMb = Debug.getNativeHeapAllocatedSize() / (1024.0 * 1024.0)
        return buildString {
            appendLine("Model  : %.1f KB (int8)".format(classifier.modelSizeBytes / 1024.0))
            appendLine("APK    : %.2f MB".format(apkSizeMb))
            appendLine("Interp : +%d KB native".format(classifier.interpreterNativeBytes / 1024))
            appendLine("Heap   : native %.1f / java %.1f MB".format(nativeMb, javaMb))
            appendLine("Invoke : %.2f ms".format(inferMs))
            appendLine("  avg %.2f  p95 %.2f  min %.2f".format(
                stats.avgInference(), stats.p95Inference(), stats.minInference()))
            appendLine("Frame  : %.1f ms (avg %.1f)".format(totalMs, stats.avgTotal()))
            appendLine("FPS    : %.1f".format(stats.fps()))
            append("Frames : ${stats.frames}")
        }
    }

    private fun logSummary() {
        Log.i(
            TAG, "frames=${stats.frames} invoke_avg=%.3fms invoke_p95=%.3fms invoke_min=%.3fms " +
                    "frame_avg=%.2fms fps=%.1f model=%.1fKB apk=%.2fMB interp_native=%dKB native_heap=%.1fMB"
                .format(
                    stats.avgInference(), stats.p95Inference(), stats.minInference(),
                    stats.avgTotal(), stats.fps(), classifier.modelSizeBytes / 1024.0, apkSizeMb,
                    classifier.interpreterNativeBytes / 1024,
                    Debug.getNativeHeapAllocatedSize() / (1024.0 * 1024.0)
                )
        )
    }

    override fun onDestroy() {
        super.onDestroy()
        // Close on the analysis thread so it never races an in-flight Invoke.
        analysisExecutor.execute { classifier.close() }
        analysisExecutor.shutdown()
    }
}
