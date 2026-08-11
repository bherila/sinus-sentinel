/// `ServerUrlNormalizer.normalize` cases covering what a real user could
/// plausibly type or paste into Settings › PHR's server URL field.
func testServerUrlNormalizer() {
    expect(ServerUrlNormalizer.normalize("") == .success(""), "empty is valid and means sync off")
    expect(ServerUrlNormalizer.normalize("   ") == .success(""), "whitespace-only is also empty")

    expect(
        ServerUrlNormalizer.normalize("phr.bherila.net") == .success("https://phr.bherila.net"),
        "a bare host gets https:// prepended"
    )
    expect(
        ServerUrlNormalizer.normalize("  phr.bherila.net  ") == .success("https://phr.bherila.net"),
        "surrounding whitespace is trimmed"
    )
    expect(
        ServerUrlNormalizer.normalize("https://phr.bherila.net/") == .success("https://phr.bherila.net"),
        "a trailing slash is stripped"
    )
    expect(
        ServerUrlNormalizer.normalize("https://phr.bherila.net///") == .success("https://phr.bherila.net"),
        "every trailing slash is stripped, not just one"
    )
    expect(
        ServerUrlNormalizer.normalize("https://x/api/phr/patients/1/sinus-settings") == .success("https://x"),
        "a deep PHR link is reduced to its server root"
    )
    expect(
        ServerUrlNormalizer.normalize("x/api/phr/patients/7/sinus-settings") == .success("https://x"),
        "a schemeless deep link still reduces to the root and gets https://"
    )

    expect(isFailure(ServerUrlNormalizer.normalize("ftp://x")), "a non-http(s) scheme must be rejected")
    expect(isFailure(ServerUrlNormalizer.normalize("https:///no-host")), "a URL with no host must be rejected")
    expect(isFailure(ServerUrlNormalizer.normalize("https://not a url")), "unparsable garbage must be rejected")
}

private func isFailure(_ outcome: ServerUrlNormalizer.Outcome) -> Bool {
    if case .failure = outcome {
        return true
    }
    return false
}
