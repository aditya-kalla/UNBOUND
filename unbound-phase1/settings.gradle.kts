pluginManagement {
    repositories {
        google()
        gradlePluginPortal()
        mavenCentral()
    }
}

dependencyResolutionManagement {
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "unbound"

include(":shared")
include(":data")
// include(":androidApp")   // added in Phase 3
// include(":desktopApp")   // added in Phase 3
