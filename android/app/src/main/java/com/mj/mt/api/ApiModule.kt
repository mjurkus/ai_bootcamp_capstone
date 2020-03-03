package com.mj.mt.api

import retrofit2.Retrofit
import retrofit2.converter.moshi.MoshiConverterFactory


object ApiModule {

    fun provideApi() : Api {
        val retrofit = Retrofit.Builder()
            .baseUrl("http://78.61.224.161:8888/api/")
            .addConverterFactory(MoshiConverterFactory.create())
            .build()

        return  retrofit.create<Api>(Api::class.java)
    }
}