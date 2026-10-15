plugins {
    id("com.android.library")
    kotlin("multiplatform")
    id("app.cash.sqldelight")
    kotlin("plugin.serialization")
}

kotlin {
    androidTarget()
    val desktop = jvm("desktop")

    sourceSets {
        val commonMain by getting {
            dependencies {
                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.9.0")
                implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.7.3")
                implementation("app.cash.sqldelight:runtime:2.0.2")
                implementation("app.cash.sqldelight:coroutines-extensions:2.0.2")
                compileOnly("net.java.dev.jna:jna:5.14.0")
            }
        }
        val commonTest by getting {
            dependencies {
                implementation(kotlin("test"))
                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.9.0")
            }
        }
        val jvmCommonMain by creating {
            dependsOn(commonMain)
            dependencies {
                compileOnly("net.java.dev.jna:jna:5.14.0")
            }
        }
        val androidMain by getting {
            dependsOn(jvmCommonMain)
            dependencies {
                implementation("app.cash.sqldelight:android-driver:2.0.2")
                implementation("net.java.dev.jna:jna:5.14.0@aar")
            }
        }
        val desktopMain by getting {
            dependsOn(jvmCommonMain)
            dependencies {
                implementation("app.cash.sqldelight:sqlite-driver:2.0.2")
                implementation("net.java.dev.jna:jna:5.14.0")
            }
        }
    }

    tasks.register<JavaExec>("scanSmoke") {
        group = "application"
        description = "Runs the desktop library scan smoke test"
        mainClass.set("com.unbound.data.scan.ScanSmokeProgramKt")
        val mainCompilation = desktop.compilations.getByName("main")
        classpath = files(
            mainCompilation.output.allOutputs,
            mainCompilation.runtimeDependencyFiles
        )
        val jnaProp = System.getProperty("jna.library.path")
        val resolvedJnaPath = if (jnaProp != null) {
            val f = file(jnaProp)
            if (f.exists()) f.absolutePath else rootProject.file(jnaProp).absolutePath
        } else if (rootProject.file("unbound_audio.dll").exists()) {
            rootProject.projectDir.absolutePath
        } else if (rootProject.file("../unbound-phase1/rust-audio-engine/target/release").exists()) {
            rootProject.file("../unbound-phase1/rust-audio-engine/target/release").absolutePath
        } else if (rootProject.file("../unbound-phase1/unbound_audio.dll").exists()) {
            rootProject.file("../unbound-phase1").absolutePath
        } else {
            rootProject.file("rust-audio-engine/target/release").absolutePath
        }
        systemProperty("jna.library.path", resolvedJnaPath)
        val scanFolder = System.getProperty("unbound.scanFolder")
        if (scanFolder != null) {
            systemProperty("unbound.scanFolder", scanFolder)
        }
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
