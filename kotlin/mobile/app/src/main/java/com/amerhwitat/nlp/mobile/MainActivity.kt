package com.amerhwitat.nlp.mobile
import android.app.Activity
import android.os.Bundle
import android.view.Gravity
import android.widget.*
class MainActivity:Activity(){override fun onCreate(savedInstanceState:Bundle?){super.onCreate(savedInstanceState);val r=LinearLayout(this).apply{orientation=LinearLayout.VERTICAL;gravity=Gravity.CENTER;setPadding(32,32,32,32)};val t=TextView(this).apply{text="Ancient NLP — Kotlin Mobile";textSize=24f;gravity=Gravity.CENTER};val s=TextView(this).apply{text="Thamudic / Safaitic / Hismaic / Dadanitic\nUnicode registry: ready\n128D perception state: ready";textSize=16f;gravity=Gravity.CENTER;setPadding(0,24,0,24)};val b=Button(this).apply{text="Open script workspace";setOnClickListener{s.text="Script workspace: active\nUTF-8 glyph model: ready\n128D perception state: active"}};r.addView(t);r.addView(s);r.addView(b);setContentView(r)}}
