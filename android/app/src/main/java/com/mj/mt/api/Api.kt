package com.mj.mt.api

import okhttp3.MultipartBody
import retrofit2.http.GET
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.Part

interface Api {

    @Multipart
    @POST("predict")
    suspend fun getPredictions(
        @Part file: MultipartBody.Part
    ) : List<Prediction>
}