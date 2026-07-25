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

# Regenerate the committed bindings from the cdylib and normalize them. Runs on
# Linux (CI's drift gate) and on a Mac (where the same command must reproduce
# byte-identical output, or the gate would only ever be satisfiable in CI) — hence
# resolving the platform's shared-library extension rather than hardcoding `.so`.
UNIFFI_LIB_EXT := $(if $(filter Darwin,$(shell uname -s)),dylib,so)
UNIFFI_LIB := target/debug/libsinus_apple.$(UNIFFI_LIB_EXT)
BINDGEN := target/debug/uniffi-bindgen-swift

apple-bindings-check:
	cargo build --locked -p sinus-apple
	cargo build --locked -p uniffi-bindgen-swift
	$(BINDGEN) --swift-sources $(UNIFFI_LIB) apps/apple/Generated/Swift
	$(BINDGEN) --headers $(UNIFFI_LIB) apps/apple/Generated/Headers
	$(BINDGEN) --modulemap --xcframework --module-name SinusAppleFFI --modulemap-filename module.modulemap $(UNIFFI_LIB) apps/apple/Generated/Modules
	./scripts/normalize-apple-bindings.sh apps/apple/Generated
