import Foundation
import Testing

@testable import EXO

struct BugReportSystemInfoTests {

    /// Answers the commands a report runs, recording them.
    private final class FakeMac {
        var commands: [[String]] = []

        func run(_ command: [String]) -> String? {
            commands.append(command)
            switch command.joined(separator: " ") {
            case "/usr/sbin/sysctl -n hw.model": return "Mac16,9"
            case "/usr/bin/sw_vers -productVersion": return "26.1"
            case "/usr/bin/sw_vers -buildVersion": return "25B78"
            case "/usr/bin/uname -m": return "arm64"
            case "/sbin/route -n get default":
                return "   route to: default\n  interface: en0\n      flags: <UP,DONE>"
            case "/usr/sbin/ipconfig getiflist": return "en0 en5 bridge0"
            case "/usr/sbin/ipconfig getifaddr en0": return "192.0.2.10"
            case "/usr/sbin/ipconfig getifaddr en5": return "169.254.10.2"
            default: return nil
            }
        }
    }

    @Test func onlyTheFieldsTriageNeedsAreSent() {
        let mac = FakeMac()

        let info = BugReportSystemInfo.collect(run: mac.run, uptime: 3600)

        #expect(Set(info.keys).isSubset(of: Set(BugReportSystemInfo.fields.keys)))
        #expect(info["hardware_model"] as? String == "Mac16,9")
        #expect(info["os_version"] as? String == "26.1")
        #expect(info["arch"] as? String == "arm64")
        #expect(info["uptime_seconds"] as? Int == 3600)
        #expect(info["default_interface"] as? String == "en0")
        let interfaces = info["interfaces"] as? [[String: String]]
        #expect(
            interfaces == [
                ["name": "en0", "ip": "192.0.2.10"], ["name": "en5", "ip": "169.254.10.2"],
            ])
    }

    @Test func nothingThatIdentifiesTheUserOrTheMacIsRead() {
        let mac = FakeMac()

        let info = BugReportSystemInfo.collect(run: mac.run, uptime: 1)

        for key in [
            "hostname", "computer_name", "local_hostname", "host_name", "hardware_profile",
            "hardware_uuid", "user", "console_user", "wifi_ssid", "default_mac", "default_ip",
        ] {
            #expect(info[key] == nil, "\(key) is sent")
        }
        let programs = Set(mac.commands.map { $0[0] })
        for program in [
            "/usr/sbin/system_profiler", "/usr/bin/whoami", "/usr/bin/stat", "/bin/hostname",
            "/usr/sbin/scutil",
        ] {
            #expect(!programs.contains(program), "\(program) is run")
        }
        #expect(!mac.commands.flatMap { $0 }.contains { $0.contains("airport") })
    }
}

struct BugReportRedactorTests {
    private let redactor = BugReportRedactor(
        homeDirectory: "/Users/jdoe",
        names: [
            (name: "Jane Doe's MacBook Pro", replacement: "<this Mac>"),
            (name: "Jane Doe", replacement: "<user>"),
            (name: "jdoe", replacement: "<user>"),
            (name: "ab", replacement: "<too short to replace>"),
        ])

    @Test func logsLoseTheAccountNameTheMacsNameAndHardwareAddresses() {
        let log = """
            [ INFO ] models in /Users/jdoe/.exo/models
            [ INFO ] node name Jane Doe's MacBook Pro, owner Jane Doe (jdoe)
            \tether 3c:22:fb:01:02:0a
            \tinet 192.0.2.10 netmask 0xffffff00 broadcast 192.0.2.255
            \tnode_guid:\t\t\t0012:3456:789a:bcde
            [ INFO ] received image task: prompt='a portrait of my family' size=1024
            [ INFO ] tab bar about
            """

        let scrubbed = redactor.scrub(log)

        #expect(
            scrubbed == """
                [ INFO ] models in ~/.exo/models
                [ INFO ] node name <this Mac>, owner <user> (<user>)
                \tether <hardware address>
                \tinet 192.0.2.10 netmask 0xffffff00 broadcast 192.0.2.255
                \tnode_guid:\t\t\t<hardware id>
                [ INFO ] received image task: <redacted>
                [ INFO ] tab bar about
                """)
    }

    @Test func stateAndEventsLosePromptsResponsesImagesAndDeviceNames() throws {
        let events = """
            [{"TaskCreated": {"task": {"TextGeneration": {"task_params": {"model": "mlx-community/Qwen3-0.6B-4bit",
              "input": [{"role": "user", "content": "my secret plan"}], "instructions": "be brief",
              "max_output_tokens": 100}}}}},
             {"ChunkGenerated": {"chunk": {"TokenChunk": {"text": "Sure", "token_id": 5}}}},
             {"InputChunkReceived": {"chunk": {"data": "iVBORw0KGgo="}}},
             {"NodeGatheredInfo": {"info": {"MiscData": {"friendly_name": "Jane Doe's MacBook Pro"}}}},
             {"NodeDownloadProgress": {"path": "/Users/jdoe/.exo/models/x", "error_message": "disk full"}}]
            """

        let scrubbed = try #require(redactor.scrubJSON(Data(events.utf8)))
        let text = String(decoding: scrubbed, as: UTF8.self)

        for secret in ["my secret plan", "be brief", "Sure", "iVBORw0KGgo", "Jane", "jdoe"] {
            #expect(!text.contains(secret), "\(secret) is sent")
        }
        // What is needed to diagnose a problem stays.
        for kept in [
            "mlx-community/Qwen3-0.6B-4bit", "\"max_output_tokens\":100", "\"token_id\":5",
            "~/.exo/models/x", "disk full", "<redacted: 1 item>", "<redacted: 8 characters>",
        ] {
            #expect(text.contains(kept), "\(kept) is missing")
        }
        // Still valid JSON.
        #expect(try JSONSerialization.jsonObject(with: scrubbed) is [Any])
    }

    @Test func promptsWrittenIntoExosLogAreTakenOut() {
        let log = """
            [ INFO ] Executing command: TextGeneration(task_params=TextGenerationTaskParams(model='m', \
            input=[InputMessage(role='user', content=<InputMessageContent: my secret > plan>)], \
            instructions=<InputMessageContent: be brief>, max_output_tokens=20, \
            chat_template_messages=[{'role': 'user', 'content': 'my "secret" plan'}, \
            {'role': 'tool', "content": "it's a secret"}], images=[]))
            [ INFO ] ImageGeneration(task_params=ImageGenerationTaskParams(prompt='a cat', size='1024x1024'))
            [ DEBUG ] TokenChunk(text='Sure', token_id=5)
            """

        let scrubbed = redactor.scrub(log)

        for secret in ["secret", "plan", "be brief", "a cat", "Sure"] {
            #expect(!scrubbed.contains(secret), "\(secret) is sent")
        }
        for kept in [
            "model='m'", "max_output_tokens=20", "size='1024x1024'", "token_id=5", "'role': 'tool'",
        ] {
            #expect(scrubbed.contains(kept), "\(kept) is missing")
        }
    }

    @Test func somethingThatIsntJSONIsNotSent() {
        #expect(redactor.scrubJSON(Data("not json".utf8)) == nil)
    }

    @Test func aReportCanBeWrittenToAFolderToLookAt() throws {
        let folder = FileManager.default.temporaryDirectory
            .appendingPathComponent("EXOBugReportTests-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: folder) }

        try BugReportService.write(
            [
                BugReportFile(path: "report.json", data: Data("{}".utf8)),
                BugReportFile(path: "exo_log/exo.log", data: Data("log".utf8)),
            ], to: folder)

        #expect(
            try String(contentsOf: folder.appendingPathComponent("report.json"), encoding: .utf8)
                == "{}")
        #expect(
            try String(
                contentsOf: folder.appendingPathComponent("exo_log/exo.log"), encoding: .utf8)
                == "log")
    }
}
