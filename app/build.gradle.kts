plugins {
    alias(libs.plugins.android.application)
    id("com.chaquo.python")
}

android {
    namespace = "com.example.music_app"
    compileSdk {
        version = release(36) {
            minorApiLevel = 1
        }
    }

    defaultConfig {
        ndk {
            abiFilters += listOf("arm64-v8a", "x86_64")
        }
        applicationId = "com.example.music_app"
        minSdk = 24
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
}

chaquopy {
    defaultConfig {
        version = "3.10"
        buildPython("C:/Users/a1679/anaconda3/envs/music_ai/python.exe")
        pip {
            install("numpy")
            install("opencv-python")
        }
    }
}
dependencies {
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.17.1")
    implementation(libs.androidx.activity.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.androidx.constraintlayout)
    implementation(libs.androidx.core.ktx)
    implementation(libs.material)
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(libs.androidx.junit)
}
