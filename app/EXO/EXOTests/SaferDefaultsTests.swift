import AppKit
import Testing

@testable import EXO

@MainActor
struct UninstallAlertTests {

    @Test func uninstallKeepsDownloadedModelsUnlessUnticked() {
        let (alert, keepModels) = UninstallEXO.makeAlert(modelsSize: nil)

        #expect(keepModels.state == .on)
        #expect(alert.accessoryView === keepModels)
        #expect(alert.buttons.map(\.title) == ["Uninstall", "Cancel"])
        #expect(alert.buttons.first?.hasDestructiveAction == true)
    }
}

struct HuggingFaceEndpointTests {

    @Test(
        arguments: [
            ("hf-mirror.com", "https://hf-mirror.com"),
            ("  hf-mirror.com/\n", "https://hf-mirror.com"),
            ("mirror.example.com:8080/hf", "https://mirror.example.com:8080/hf"),
            ("localhost:8080", "https://localhost:8080"),
            ("//hf-mirror.com", "https://hf-mirror.com"),
        ])
    func addsHTTPSWhenTheSchemeIsMissing(_ typed: String, _ expected: String) {
        #expect(HuggingFaceEndpoint.normalized(typed) == expected)
    }

    @Test(
        arguments: [
            ("https://hf-mirror.com", "https://hf-mirror.com"),
            ("http://mirror.example.com:8080", "http://mirror.example.com:8080"),
            ("https://hf-mirror.com/", "https://hf-mirror.com"),
            ("HTTPS://hf-mirror.com", "HTTPS://hf-mirror.com"),
        ])
    func keepsAGivenScheme(_ typed: String, _ expected: String) {
        #expect(HuggingFaceEndpoint.normalized(typed) == expected)
    }

    @Test(arguments: ["", "   ", "/"])
    func emptyMeansTheDefaultEndpoint(_ typed: String) {
        #expect(HuggingFaceEndpoint.normalized(typed) == "")
    }

    @Test func normalisingTwiceChangesNothing() {
        let once = HuggingFaceEndpoint.normalized("hf-mirror.com/")
        #expect(HuggingFaceEndpoint.normalized(once) == once)
    }
}
