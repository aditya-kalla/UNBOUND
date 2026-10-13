// STARTING POINT - Antigravity (Prompt 1, task A) must make this build on your setup
// (Gradle 8.9, JDK 21, Kotlin 2.0.21, AGP 8.5.2).
plugins {
    id("com.android.library")
    kotlin("multiplatform")
    id("app.cash.sqldelight")
}

kotlin {
    androidTarget()
    jvm("desktop")

    sourceSets {
        val commonMain by getting {
            dependencies {
                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.9.0")
                implementation("app.cash.sqldelight:runtime:2.0.2")
                implementation("app.cash.sqldelight:coroutines-extensions:2.0.2")
            }
        }
        val commonTest by getting {
            dependencies {
                implementation(kotlin("test"))
                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.9.0")
            }
        }
        val androidMain by getting { dependencies { implementation("app.cash.sqldelight:android-driver:2.0.2") } }
        val desktopMain by getting { dependencies { implementation("app.cash.sqldelight:sqlite-driver:2.0.2") } }
    }
}

sqldelight {
    databases {
        create("UnboundDatabase") {
            packageName.set("com.unbound.data.db")
            dialect("app.cash.sqldelight:sqlite-3-38-dialect:2.0.2")
            // verifyMigrations.set(true)   // turn on with the first .sqm migration
        }
    }
}

android {
    namespace = "com.unbound.data"
    compileSdk = 35
    defaultConfig { minSdk = 26 }
}
