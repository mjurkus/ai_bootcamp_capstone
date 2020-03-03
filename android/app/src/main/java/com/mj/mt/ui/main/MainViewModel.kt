package com.mj.mt.ui.main

import android.net.Uri
import android.util.Log
import androidx.lifecycle.ViewModel
import com.mj.mt.api.Api
import com.mj.mt.api.ApiModule
import kotlinx.coroutines.launch
import okhttp3.MediaType
import okhttp3.MultipartBody
import okhttp3.RequestBody
import java.io.File

class MainViewModel : BaseViewModel() {

    private val api : Api = ApiModule.provideApi()

    fun getPredictions(imageFilePath : String) {
        Log.d(TAG, "Received $imageFilePath}")

        launch {
            val file = File(imageFilePath)
            val body = RequestBody.create(MediaType.parse("image/jpeg"), file)
            val predictions = api.getPredictions(
                file = MultipartBody.Part.createFormData("file", file.name, body)
            )

            Log.d(TAG, "${predictions}")
        }
    }

    companion object {
        const val TAG = "MainViewModel"
    }
}
