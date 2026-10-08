package com.edgeai.signedge

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import kotlin.math.min

/**
 * Draws the square region of interest that is cropped from each camera frame and fed to the
 * model. Assumes the PreviewView uses FIT_CENTER, so the whole (upright) camera image is
 * visible and letterboxed inside this view.
 */
class RoiOverlayView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null,
) : View(context, attrs) {

    private val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 6f
        color = Color.WHITE
    }
    private val rect = RectF()

    /** Upright camera image size (after rotation) used to place the ROI. */
    private var imageW = 3
    private var imageH = 4

    fun setImageSize(w: Int, h: Int) {
        if (w != imageW || h != imageH) {
            imageW = w; imageH = h
            postInvalidate()
        }
    }

    fun setAccepted(accepted: Boolean) {
        val c = if (accepted) Color.GREEN else Color.WHITE
        if (paint.color != c) {
            paint.color = c
            postInvalidate()
        }
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val scale = min(width / imageW.toFloat(), height / imageH.toFloat())
        val dispW = imageW * scale
        val dispH = imageH * scale
        val side = min(dispW, dispH) * MainActivity.ROI_FRACTION
        val cx = width / 2f
        val cy = height / 2f
        rect.set(cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)
        canvas.drawRect(rect, paint)
    }
}
