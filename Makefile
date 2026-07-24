.PHONY: apple-ios-run apple-ios-build apple-macos-run apple-macos-build apple-bindings-check

# Builds the Rust dependency, regenerates Swift bindings, compiles an app bundle,
# boots/selects an iOS Simulator, installs the app, and launches it.
apple-ios-run:
	./scripts/apple-dev.sh ios

apple-ios-build:
	./scripts/apple-dev.sh ios build

# Builds the same Rust dependency and launches the native macOS SwiftUI shell.
apple-macos-run:
	./scripts/apple-dev.sh macos

apple-macos-build:
	./scripts/apple-dev.sh macos build

# Linux-safe contract check used before handing the branch to a Mac.
apple-bindings-check:
	cargo build --locked -p sinus-apple
	cargo build --locked -p uniffi-bindgen-swift
	target/debug/uniffi-bindgen-swift --swift-sources target/debug/libsinus_apple.so apps/apple/Generated/Swift
	target/debug/uniffi-bindgen-swift --headers target/debug/libsinus_apple.so apps/apple/Generated/Headers
	target/debug/uniffi-bindgen-swift --modulemap --xcframework --module-name SinusAppleFFI --modulemap-filename module.modulemap target/debug/libsinus_apple.so apps/apple/Generated/Modules
	./scripts/normalize-apple-bindings.sh apps/apple/Generated
