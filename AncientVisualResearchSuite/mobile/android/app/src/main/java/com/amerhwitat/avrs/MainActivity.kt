package com.amerhwitat.avrs

import android.app.Activity
import android.os.Bundle
import android.widget.TextView

class MainActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(TextView(this).apply {
            text = "Ancient Visual Research Suite\nHistorical Event Visualization"
            textSize = 20f
            setPadding(32, 48, 32, 32)
        })
    }
}
