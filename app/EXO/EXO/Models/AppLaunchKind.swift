import AppKit
import Foundation

/// How the app was opened, which decides whether it may open anything on its
/// own: a login item starts silently.
enum AppLaunchKind: Equatable {
    /// Opened by the user (Finder, Spotlight, `open`), or relaunched after an
    /// update.
    case manual
    /// Opened by macOS as a login item.
    case atLogin

    /// Reads the open-application Apple Event that launched the app. macOS
    /// marks the launch of a login item, including one registered with
    /// `SMAppService.mainApp`, with `keyAELaunchedAsLogInItem`.
    static func of(eventID: AEEventID?, launchProperty: OSType?) -> AppLaunchKind {
        guard eventID == AEEventID(kAEOpenApplication),
            launchProperty == OSType(keyAELaunchedAsLogInItem)
        else {
            return .manual
        }
        return .atLogin
    }

    /// The launch the app is handling now. Only meaningful while the
    /// open-application event is being handled, i.e. in
    /// `applicationDidFinishLaunching(_:)`.
    static func ofCurrentAppleEvent() -> AppLaunchKind {
        let event = NSAppleEventManager.shared().currentAppleEvent
        return of(
            eventID: event?.eventID,
            launchProperty: event?.paramDescriptor(forKeyword: keyAEPropData)?.enumCodeValue)
    }
}
