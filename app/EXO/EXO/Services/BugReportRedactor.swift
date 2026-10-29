import Foundation
import SystemConfiguration

/// Takes what isn't needed to diagnose a problem out of a bug report: who the
/// user is, what they asked their models, and hardware addresses.
struct BugReportRedactor {
    /// Keys in exo's state and events whose values are what people sent to
    /// their models or got back (prompts, messages, generated text, images),
    /// or a device's name, which is often its owner's name. Their values are
    /// replaced by a note of their size, which is what a bug usually needs.
    static let redactedKeys: Set<String> = [
        "chat_template_messages", "content", "data", "friendly_name", "image_data", "images",
        "input", "instructions", "messages", "prompt", "text", "tool_calls", "top_logprobs",
        "user",
    ]

    /// The home folder, shown as ~ so paths don't carry the account name.
    var homeDirectory: String
    /// Names that identify the user or this Mac, each with what replaces it.
    var names: [(name: String, replacement: String)]

    /// The redactor for this Mac and user.
    static var current: BugReportRedactor {
        var names: [(String, String)] = []
        if let computer = SCDynamicStoreCopyComputerName(nil, nil) as String? {
            names.append((computer, "<this Mac>"))
        }
        if let local = SCDynamicStoreCopyLocalHostName(nil) as String? {
            names.append((local, "<this Mac>"))
        }
        names.append((NSFullUserName(), "<user>"))
        names.append((NSUserName(), "<user>"))
        return BugReportRedactor(homeDirectory: NSHomeDirectory(), names: names)
    }

    init(homeDirectory: String, names: [(name: String, replacement: String)]) {
        self.homeDirectory = homeDirectory
        // Longest first, so a computer name containing the user's name is
        // replaced whole. Very short names would match inside other words.
        self.names = names.filter { $0.name.count >= 3 }.sorted { $0.name.count > $1.name.count }
    }

    /// Text with the home folder, the user's and this Mac's names, hardware
    /// addresses and image prompts in exo's log taken out.
    func scrub(_ text: String) -> String {
        var text = text
        if homeDirectory.count > 1 {
            text = text.replacingOccurrences(of: homeDirectory, with: "~")
        }
        for (name, replacement) in names {
            text = text.replacingOccurrences(
                of: "\\b\(NSRegularExpression.escapedPattern(for: name))\\b", with: replacement,
                options: [.regularExpression, .caseInsensitive])
        }
        for (pattern, replacement) in Self.textPatterns {
            text = text.replacingOccurrences(
                of: pattern, with: replacement, options: [.regularExpression, .caseInsensitive])
        }
        return text
    }

    private static let textPatterns: [(String, String)] = [
        // MAC addresses (ifconfig's "ether", logs).
        ("\\b[0-9a-f]{2}(:[0-9a-f]{2}){5}\\b", "<hardware address>"),
        // RDMA node and system image GUIDs (ibv_devinfo).
        ("((?:node|sys_image)_guid:\\s*)[0-9a-f:.]+", "$1<hardware id>"),
        // exo logs the start of an image task, prompt included.
        ("(received image task: )[^\\n]*", "$1<redacted>"),
        // exo logs each request with its messages, cut to 100 characters.
        (
            "<InputMessageContent: [\\s\\S]{0,110}?>(?=[)\\]},])",
            "<InputMessageContent: <redacted>>"
        ),
        // Messages, prompts and generated text written out as Python values.
        (
            "(['\"]content['\"]: |\\b(?:prompt|instructions|text)=)"
                + "(?:'(?:[^'\\\\\\n]|\\\\.)*'|\"(?:[^\"\\\\\\n]|\\\\.)*\")",
            "$1'<redacted>'"
        ),
    ]

    /// exo's JSON (state or events) with `redactedKeys` and the text in it
    /// scrubbed, or nil if it isn't JSON, in which case it can't be checked
    /// and isn't sent.
    func scrubJSON(_ data: Data) -> Data? {
        guard
            let object = try? JSONSerialization.jsonObject(
                with: data, options: [.fragmentsAllowed]),
            let redacted = try? JSONSerialization.data(
                withJSONObject: Self.redactingKeys(in: object),
                options: [.fragmentsAllowed, .sortedKeys, .withoutEscapingSlashes])
        else { return nil }
        // The rest is scrubbed in the JSON text at once, which is much faster
        // than string by string for thousands of events. What replaces a
        // name or path has no quotes or backslashes, so the JSON stays valid.
        return Data(scrub(String(decoding: redacted, as: UTF8.self)).utf8)
    }

    private static func redactingKeys(in value: Any) -> Any {
        switch value {
        case let dictionary as [String: Any]:
            var result: [String: Any] = [:]
            for (key, child) in dictionary {
                result[key] =
                    redactedKeys.contains(key) ? sizeNote(for: child) : redactingKeys(in: child)
            }
            return result
        case let array as [Any]:
            return array.map { redactingKeys(in: $0) }
        default:
            return value
        }
    }

    private static func sizeNote(for value: Any) -> Any {
        switch value {
        case let string as String:
            return string.isEmpty ? string : "<redacted: \(counted(string.count, "character"))>"
        case let array as [Any]:
            return "<redacted: \(counted(array.count, "item"))>"
        case let dictionary as [String: Any]:
            return "<redacted: \(counted(dictionary.count, "field"))>"
        case is NSNull:
            return value
        default:
            return "<redacted>"
        }
    }

    private static func counted(_ count: Int, _ noun: String) -> String {
        "\(count) \(noun)\(count == 1 ? "" : "s")"
    }
}
