import Foundation

/// The HuggingFace endpoint setting, as exo needs it in HF_ENDPOINT.
enum HuggingFaceEndpoint {
    /// exo builds download URLs as "<HF_ENDPOINT>/<path>", so a value
    /// without a scheme, like "hf-mirror.com", breaks every download. Adds
    /// "https://" when the scheme is missing and drops surrounding whitespace
    /// and trailing slashes. Empty stays empty, which means huggingface.co.
    static func normalized(_ endpoint: String) -> String {
        var value = endpoint.trimmingCharacters(in: .whitespacesAndNewlines)
        while value.hasSuffix("/") {
            value.removeLast()
        }
        guard !value.isEmpty else { return "" }
        if value.range(of: "^[A-Za-z][A-Za-z0-9+.-]*://", options: .regularExpression) == nil {
            while value.hasPrefix("/") {
                value.removeFirst()
            }
            value = "https://" + value
        }
        return value
    }
}
