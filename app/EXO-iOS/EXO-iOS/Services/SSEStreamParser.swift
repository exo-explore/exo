import Foundation

struct SSEStreamParser<T: Decodable>: AsyncSequence {
    typealias Element = T

    let bytes: URLSession.AsyncBytes
    let decoder: JSONDecoder

    init(bytes: URLSession.AsyncBytes, decoder: JSONDecoder = JSONDecoder()) {
        self.bytes = bytes
        self.decoder = decoder
    }

    func makeAsyncIterator() -> AsyncIterator {
        AsyncIterator(lines: bytes.lines, decoder: decoder)
    }

    struct AsyncIterator: AsyncIteratorProtocol {
        var lines: AsyncLineSequence<URLSession.AsyncBytes>.AsyncIterator
        let decoder: JSONDecoder

        init(lines: AsyncLineSequence<URLSession.AsyncBytes>, decoder: JSONDecoder) {
            self.lines = lines.makeAsyncIterator()
            self.decoder = decoder
        }

        mutating func next() async throws -> T? {
            while let line = try await lines.next() {
                let trimmed = line.trimmingCharacters(in: .whitespaces)

                guard trimmed.hasPrefix("data: ") else { continue }

                let payload = String(trimmed.dropFirst(6))

                if payload == "[DONE]" {
                    return nil
                }

                guard let data = payload.data(using: .utf8) else { continue }

                do {
                    return try decoder.decode(T.self, from: data)
                } catch {
                    continue
                }
            }
            return nil
        }
    }
}
