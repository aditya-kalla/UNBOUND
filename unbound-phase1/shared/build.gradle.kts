plugins {
    id("com.android.library")
    kotlin("multiplatform")
}

kotlin {
    androidTarget()
    val desktop = jvm("desktop") // Windows / macOS / Linux host

    sourceSets {
        val commonMain by getting {
            dependencies {
                compileOnly("net.java.dev.jna:jna:5.14.0")
            }
        }
        val commonTest by getting {
            dependencies {
                implementation(kotlin("test"))
            }
        }
        val androidMain by getting {
            dependencies {
                implementation("net.java.dev.jna:jna:5.14.0@aar")
            }
        }
        val desktopMain by getting {
            dependencies {
                implementation("net.java.dev.jna:jna:5.14.0")
            }
        }
    }

    tasks.register<JavaExec>("desktopRun") {
        group = "application"
        description = "Runs the desktop smoke test"
        mainClass.set("com.unbound.audio.DesktopSmokeTestKt")
        val mainCompilation = desktop.compilations.getByName("main")
        classpath = files(
            mainCompilation.output.allOutputs,
            mainCompilation.runtimeDependencyFiles
        )
        val defaultRustReleaseDir = rootProject.file("rust-audio-engine/target/release").absolutePath
        val jnaProp = System.getProperty("jna.library.path")
        val resolvedJnaPath = if (jnaProp != null) {
            val f = file(jnaProp)
            if (f.exists()) f.absolutePath else rootProject.file(jnaProp).absolutePath
        } else {
            defaultRustReleaseDir
        }
        systemProperty("jna.library.path", resolvedJnaPath)
        val testFile = System.getProperty("unbound.testFile")
        if (testFile != null) {
            systemProperty("unbound.testFile", testFile)
        }
    }
}

android {
    namespace = "com.unbound.audio"
    compileSdk = 35
    defaultConfig {
        minSdk = 26
    }
}
