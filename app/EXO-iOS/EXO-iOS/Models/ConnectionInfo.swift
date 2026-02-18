import Foundation

struct ConnectionInfo: Codable, Equatable {
    let host: String
    let port: Int
    let nodeId: String?

    var baseURL: URL { URL(string: "http://\(host):\(port)")! }

    static let defaultPort = 52415
}
