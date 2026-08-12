import Foundation

/// `DevicePairing`'s pure pieces — the request-building/response-parsing half
/// of "Sign in with PHR" that does not need `AuthenticationServices` or a
/// live network round trip to exercise. See `Models/DevicePairing.swift` for
/// the FROZEN wire contract this implements.
func testDevicePairing() {
    testChallenge()
    testMakeRequest()
    testCodeFromCallback()
    testParseExchangeResponse()
    testGenerateVerifier()
}

/// Independent derivation (see the doc comment on `challenge(for:)` for the
/// shell/Python steps): `"test-verifier"` -> SHA-256 hex
/// `2416e2a8e34658f6809b05e4ffc6d3e949e53dfae7eb90f7d9e665252fb3186` ->
/// base64 `JBbiqONGWPaAmwXk/8bT6UnlPfrn65D32eZlJS+zGG0=` -> base64url (no
/// padding) `JBbiqONGWPaAmwXk_8bT6UnlPfrn65D32eZlJS-zGG0`.
private func testChallenge() {
    expect(
        DevicePairing.challenge(for: "test-verifier") == "JBbiqONGWPaAmwXk_8bT6UnlPfrn65D32eZlJS-zGG0",
        "challenge(for:) matches an independently-derived base64url SHA-256"
    )
    // No stray `+`, `/`, or `=` should ever survive the base64url transform.
    let challenge = DevicePairing.challenge(for: "another-verifier-entirely")
    expect(!challenge.contains("+"), "challenge never contains '+'")
    expect(!challenge.contains("/"), "challenge never contains '/'")
    expect(!challenge.contains("="), "challenge never contains padding")
}

private func testMakeRequest() {
    expect(
        DevicePairing.makeRequest(serverRoot: "", deviceId: "d1", deviceName: "Mac") == nil,
        "empty serverRoot yields nil"
    )

    guard let request = DevicePairing.makeRequest(
        serverRoot: "https://phr.bherila.net",
        deviceId: "device-123",
        deviceName: "Ben's Mac Studio",
        randomVerifier: "fixed-verifier-for-test"
    ) else {
        expect(false, "makeRequest should succeed for a valid serverRoot")
        return
    }

    expect(request.verifier == "fixed-verifier-for-test", "the injected verifier is used verbatim")

    guard let components = URLComponents(url: request.url, resolvingAgainstBaseURL: false) else {
        expect(false, "the built URL must itself be parsable")
        return
    }
    expect(components.scheme == "https", "scheme is preserved")
    expect(components.host == "phr.bherila.net", "host is preserved")
    expect(components.path == "/device-pairing", "path is exactly /device-pairing")

    let items = components.queryItems ?? []
    expect(items.count == 4, "exactly four query params, got \(items.count)")

    func value(_ name: String) -> String? {
        items.first(where: { $0.name == name })?.value
    }
    expect(value("device_id") == "device-123", "device_id passed through")
    expect(value("name") == "Ben's Mac Studio", "device name with a space and apostrophe round-trips through decoding")
    expect(
        value("code_challenge") == DevicePairing.challenge(for: "fixed-verifier-for-test"),
        "code_challenge matches challenge(for:) on the same verifier"
    )
    expect(value("redirect_uri") == "sinussentinel://paired", "redirect_uri is the fixed callback URL")

    // The raw URL string itself must not contain a literal space — proof the
    // encoding actually happened, not just that decoding it back out worked.
    expect(!request.url.absoluteString.contains(" "), "the device name's space is percent-encoded in the raw URL")

    // A server URL stored before ServerUrlNormalizer existed may still end in
    // a slash; the path must come out /device-pairing, not //device-pairing.
    if let legacy = DevicePairing.makeRequest(
        serverRoot: "https://phr.bherila.net/",
        deviceId: "d1",
        deviceName: "Mac",
        randomVerifier: "v"
    ) {
        expect(
            URLComponents(url: legacy.url, resolvingAgainstBaseURL: false)?.path == "/device-pairing",
            "a trailing slash on the stored root does not double the path separator"
        )
    } else {
        expect(false, "a trailing-slash serverRoot must still build a request")
    }
}

private func testCodeFromCallback() {
    if case .code(let code) = DevicePairing.code(fromCallback: URL(string: "sinussentinel://paired?code=abc123")!) {
        expect(code == "abc123", "the code param is extracted verbatim")
    } else {
        expect(false, "a code= callback should decode as .code")
    }

    expect(
        DevicePairing.code(fromCallback: URL(string: "sinussentinel://paired?error=denied")!) == .denied,
        "error=denied decodes as .denied"
    )

    expect(
        DevicePairing.code(fromCallback: URL(string: "sinussentinel://paired")!) == .malformed,
        "neither code nor error present is .malformed"
    )
    expect(
        DevicePairing.code(fromCallback: URL(string: "sinussentinel://paired?error=something_else")!) == .malformed,
        "an error value other than denied is .malformed, not silently treated as denied"
    )
    expect(
        DevicePairing.code(fromCallback: URL(string: "sinussentinel://paired?code=")!) == .malformed,
        "an empty code value is .malformed, not an empty success"
    )
}

private func testParseExchangeResponse() {
    let happyBody = Data("""
    {"token":"tok_abc","expires_at":"2026-09-01T00:00:00Z","device_name":"Ben's Mac Studio"}
    """.utf8)
    if case .success(let success) = DevicePairing.parseExchangeResponse(status: 200, data: happyBody) {
        expect(success.token == "tok_abc", "token is extracted")
        expect(success.expiresAt == "2026-09-01T00:00:00Z", "expiresAt is extracted")
        expect(success.deviceName == "Ben's Mac Studio", "deviceName is extracted")
    } else {
        expect(false, "a well-formed 200 body should parse as .success")
    }

    let tolerantBody = Data("""
    {"token":"tok_xyz","expires_at":"2026-09-01T00:00:00Z","device_name":"Mac","unexpected_field":42,"nested":{"a":1}}
    """.utf8)
    if case .success(let success) = DevicePairing.parseExchangeResponse(status: 200, data: tolerantBody) {
        expect(success.token == "tok_xyz", "unknown extra JSON fields do not prevent decoding the known ones")
    } else {
        expect(false, "unknown extra fields must be tolerated, not rejected")
    }

    expect(
        DevicePairing.parseExchangeResponse(
            status: 400,
            data: Data(#"{"message":"Invalid or expired pairing code."}"#.utf8)
        ) == .invalidCode,
        "400 maps to .invalidCode regardless of body detail"
    )
    expect(
        DevicePairing.parseExchangeResponse(status: 429, data: Data()) == .throttled,
        "429 maps to .throttled"
    )

    let garbageBody = Data("not json at all".utf8)
    guard case .serverError = DevicePairing.parseExchangeResponse(status: 200, data: garbageBody) else {
        expect(false, "a 200 with an undecodable body must be .serverError, not a crash or false success")
        return
    }

    guard case .serverError = DevicePairing.parseExchangeResponse(status: 422, data: Data()) else {
        expect(false, "422 (malformed shape) falls through to .serverError")
        return
    }
}

private func testGenerateVerifier() {
    let alphabet = Set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
    let first = DevicePairing.generateVerifier()
    let second = DevicePairing.generateVerifier()

    expect(first.count == 96, "generateVerifier produces exactly 96 characters")
    expect(second.count == 96, "every call produces exactly 96 characters")
    expect(first.allSatisfy { alphabet.contains($0) }, "every character is drawn from the alphanumeric alphabet")
    expect(second.allSatisfy { alphabet.contains($0) }, "every character is drawn from the alphanumeric alphabet")
    expect(first != second, "two calls produce different verifiers")
}
