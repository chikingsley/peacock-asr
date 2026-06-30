import Foundation

struct QwenByteLevelTokenizer {
    private struct TokenizerJSON: Decodable {
        struct Model: Decodable {
            let vocab: [String: Int]
        }

        struct AddedToken: Decodable {
            let id: Int
            let content: String
            let special: Bool
        }

        let model: Model
        let addedTokens: [AddedToken]

        enum CodingKeys: String, CodingKey {
            case model
            case addedTokens = "added_tokens"
        }
    }

    private struct Token {
        let content: String
        let special: Bool
    }

    private let tokensById: [Int: Token]
    private let byteByScalar: [UnicodeScalar: UInt8]

    init(tokenizerJSON: URL) throws {
        let data = try Data(contentsOf: tokenizerJSON)
        let decoded = try JSONDecoder().decode(TokenizerJSON.self, from: data)
        var tokens: [Int: Token] = [:]
        for (content, id) in decoded.model.vocab {
            tokens[id] = Token(content: content, special: false)
        }
        for addedToken in decoded.addedTokens {
            tokens[addedToken.id] = Token(
                content: addedToken.content,
                special: addedToken.special
            )
        }
        self.tokensById = tokens
        self.byteByScalar = Self.makeByteDecoder()
    }

    func decode(_ ids: [Int], skipSpecialTokens: Bool = true) -> String {
        var encoded = ""
        for id in ids {
            guard let token = tokensById[id] else { continue }
            if skipSpecialTokens && token.special {
                continue
            }
            encoded += token.content
        }

        var bytes: [UInt8] = []
        bytes.reserveCapacity(encoded.unicodeScalars.count)
        for scalar in encoded.unicodeScalars {
            if let byte = byteByScalar[scalar] {
                bytes.append(byte)
            } else if scalar.value <= UInt8.max {
                bytes.append(UInt8(scalar.value))
            }
        }
        return String(decoding: bytes, as: UTF8.self)
    }

    private static func makeByteDecoder() -> [UnicodeScalar: UInt8] {
        var bytes = Array(UInt8(ascii: "!")...UInt8(ascii: "~"))
        bytes.append(contentsOf: UInt8(0xA1)...UInt8(0xAC))
        bytes.append(contentsOf: UInt8(0xAE)...UInt8(0xFF))

        var codePoints = bytes.map { Int($0) }
        var extra = 0
        for byte in UInt8.min...UInt8.max {
            if !bytes.contains(byte) {
                bytes.append(byte)
                codePoints.append(256 + extra)
                extra += 1
            }
        }

        var result: [UnicodeScalar: UInt8] = [:]
        for (byte, codePoint) in zip(bytes, codePoints) {
            if let scalar = UnicodeScalar(codePoint) {
                result[scalar] = byte
            }
        }
        return result
    }
}

func normalizedTranscript(_ text: String) -> String {
    let lowered = text.lowercased()
    let scalars = lowered.unicodeScalars.map { scalar -> Character in
        if CharacterSet.alphanumerics.contains(scalar) || CharacterSet.whitespacesAndNewlines.contains(scalar) {
            return Character(scalar)
        }
        return " "
    }
    return String(scalars)
        .split(whereSeparator: { $0.isWhitespace })
        .joined(separator: " ")
}

func editDistance<T: Equatable>(_ left: [T], _ right: [T]) -> Int {
    if left.isEmpty { return right.count }
    if right.isEmpty { return left.count }

    var previous = Array(0...right.count)
    var current = [Int](repeating: 0, count: right.count + 1)
    for leftIndex in 1...left.count {
        current[0] = leftIndex
        for rightIndex in 1...right.count {
            if left[leftIndex - 1] == right[rightIndex - 1] {
                current[rightIndex] = previous[rightIndex - 1]
            } else {
                current[rightIndex] = min(
                    previous[rightIndex] + 1,
                    current[rightIndex - 1] + 1,
                    previous[rightIndex - 1] + 1
                )
            }
        }
        swap(&previous, &current)
    }
    return previous[right.count]
}

func wordErrorRate(reference: String, hypothesis: String) -> Double {
    let referenceWords = reference.split(separator: " ").map(String.init)
    let hypothesisWords = hypothesis.split(separator: " ").map(String.init)
    guard !referenceWords.isEmpty else {
        return hypothesisWords.isEmpty ? 0.0 : 1.0
    }
    return Double(editDistance(referenceWords, hypothesisWords)) / Double(referenceWords.count)
}

func characterErrorRate(reference: String, hypothesis: String) -> Double {
    let referenceCharacters = Array(reference)
    let hypothesisCharacters = Array(hypothesis)
    guard !referenceCharacters.isEmpty else {
        return hypothesisCharacters.isEmpty ? 0.0 : 1.0
    }
    return Double(editDistance(referenceCharacters, hypothesisCharacters)) / Double(referenceCharacters.count)
}
