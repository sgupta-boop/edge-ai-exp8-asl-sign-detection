package com.edgeai.signedge

import android.content.Context
import android.graphics.Bitmap
import android.os.Debug
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.roundToInt

const val TAG = "EdgeProfile"

data class Prediction(
    val label: String,
    val index: Int,
    val confidence: Float,
    val probs: FloatArray,
    val inferenceMs: Double,
)

/**
 * Android counterpart of the TFLite-Micro lifecycle in the lab sheet:
 *   model_data.h in flash      -> memory-mapped, uncompressed .tflite asset
 *   MicroInterpreter + arena   -> Interpreter (tensors allocated in the constructor)
 *   input(0) / Invoke / output -> int8 input buffer / run() / int8 output buffer
 */
class SignClassifier(context: Context, numThreads: Int = 2) {

    val labels: List<String> = context.assets.open("labels.txt").bufferedReader()
        .readLines().map { it.trim() }.filter { it.isNotEmpty() }

    val modelSizeBytes: Long
    /** Native heap growth caused by creating the interpreter (≈ tensor arena + runtime). */
    val interpreterNativeBytes: Long

    private val interpreter: Interpreter
    val inputSize: Int
    private val inScale: Float
    private val inZero: Int
    private val outScale: Float
    private val outZero: Int

    private val inputBuf: ByteBuffer
    private val outputBuf: Array<ByteArray>
    private val pixels: IntArray

    init {
        val model = loadModel(context, MODEL_FILE)
        modelSizeBytes = model.capacity().toLong()

        val before = Debug.getNativeHeapAllocatedSize()
        interpreter = Interpreter(model, Interpreter.Options().setNumThreads(numThreads))
        interpreter.allocateTensors()
        interpreterNativeBytes = Debug.getNativeHeapAllocatedSize() - before

        val inT = interpreter.getInputTensor(0)
        val outT = interpreter.getOutputTensor(0)
        inputSize = inT.shape()[1]
        inScale = inT.quantizationParams().scale
        inZero = inT.quantizationParams().zeroPoint
        outScale = outT.quantizationParams().scale
        outZero = outT.quantizationParams().zeroPoint

        inputBuf = ByteBuffer.allocateDirect(inputSize * inputSize).order(ByteOrder.nativeOrder())
        outputBuf = arrayOf(ByteArray(outT.shape()[1]))
        pixels = IntArray(inputSize * inputSize)

        Log.i(TAG, "Model ${modelSizeBytes / 1024.0} KB, input ${inT.shape().contentToString()} " +
                "${inT.dataType()} (s=$inScale z=$inZero), output ${outT.shape().contentToString()} " +
                "(s=$outScale z=$outZero), interpreter native heap +${interpreterNativeBytes / 1024} KB")
    }

    /** [bitmap] must already be the inputSize x inputSize ROI. */
    fun classify(bitmap: Bitmap): Prediction {
        bitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)
        inputBuf.rewind()
        for (p in pixels) {
            val r = (p shr 16) and 0xFF
            val g = (p shr 8) and 0xFF
            val b = p and 0xFF
            // Same luma formula as cv2.COLOR_BGR2GRAY used in training.
            val gray = (0.299f * r + 0.587f * g + 0.114f * b) / 255f
            val q = (gray / inScale + inZero).roundToInt().coerceIn(-128, 127)
            inputBuf.put(q.toByte())
        }
        inputBuf.rewind()

        val t0 = SystemClock.elapsedRealtimeNanos()
        interpreter.run(inputBuf, outputBuf)
        val inferenceMs = (SystemClock.elapsedRealtimeNanos() - t0) / 1e6

        val out = outputBuf[0]
        // The model ends in softmax, so dequantized outputs are already probabilities;
        // renormalise to absorb quantisation rounding.
        val probs = FloatArray(out.size) { ((out[it] - outZero) * outScale).coerceAtLeast(0f) }
        val sum = probs.sum().takeIf { it > 0f } ?: 1f
        for (i in probs.indices) probs[i] /= sum
        val best = probs.indices.maxBy { probs[it] }
        return Prediction(labels[best], best, probs[best], probs, inferenceMs)
    }

    fun close() = interpreter.close()

    private fun loadModel(context: Context, name: String): MappedByteBuffer {
        context.assets.openFd(name).use { fd ->
            FileInputStream(fd.fileDescriptor).channel.use { ch ->
                return ch.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
            }
        }
    }

    companion object {
        const val MODEL_FILE = "asl_int8.tflite"
    }
}

/** Rolling latency statistics (avg / p95 / fps) over the last [window] frames. */
class LatencyStats(private val window: Int = 300) {
    private val inference = ArrayDeque<Double>()
    private val total = ArrayDeque<Double>()
    private val frameTimes = ArrayDeque<Long>()
    var frames = 0L
        private set

    @Synchronized
    fun add(inferenceMs: Double, totalMs: Double) {
        frames++
        push(inference, inferenceMs)
        push(total, totalMs)
        frameTimes.addLast(SystemClock.elapsedRealtime())
        if (frameTimes.size > window) frameTimes.removeFirst()
    }

    private fun push(q: ArrayDeque<Double>, v: Double) {
        q.addLast(v)
        if (q.size > window) q.removeFirst()
    }

    @Synchronized
    fun reset() {
        inference.clear(); total.clear(); frameTimes.clear(); frames = 0
    }

    @Synchronized fun avgInference() = inference.averageOrZero()
    @Synchronized fun p95Inference() = inference.percentile(0.95)
    @Synchronized fun minInference() = inference.minOrNull() ?: 0.0
    @Synchronized fun avgTotal() = total.averageOrZero()

    @Synchronized
    fun fps(): Double {
        if (frameTimes.size < 2) return 0.0
        val span = (frameTimes.last() - frameTimes.first()) / 1000.0
        return if (span > 0) (frameTimes.size - 1) / span else 0.0
    }

    private fun ArrayDeque<Double>.averageOrZero() = if (isEmpty()) 0.0 else average()
    private fun ArrayDeque<Double>.percentile(p: Double): Double {
        if (isEmpty()) return 0.0
        val s = sorted()
        return s[((s.size - 1) * p).toInt()]
    }
}
