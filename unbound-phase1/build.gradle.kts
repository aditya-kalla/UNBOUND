plugins {
    // Versions pinned centrally; bump deliberately, not silently.
    kotlin("multiplatform") version "2.0.21" apply false
    kotlin("android") version "2.0.21" apply false
    id("com.android.application") version "8.5.2" apply false
    id("com.android.library") version "8.5.2" apply false
    id("org.jetbrains.compose") version "1.7.0" apply false
    id("app.cash.sqldelight") version "2.0.2" apply false
    kotlin("plugin.serialization") version "2.0.21" apply false
}

allprojects {
    repositories {
        google()
        mavenCentral()
    }
}

tasks.register<Exec>("buildRustEngine") {
    group = "unbound"
    description = "Builds the Rust audio engine for all configured targets and copies the resulting " +
        "shared libraries into shared/src/*/resources so the JVM/Android side can load them."
    workingDir = file("rust-audio-engine")
    commandLine("cargo", "build", "--release")
    doLast {
        // Phase 1: desktop-host build only. Android cross-compilation targets
        // (aarch64-linux-android, etc.) are added once the Android app module lands.
        println("Rust engine built. Wire target-specific copy steps here as platforms are added.")
    }
}
