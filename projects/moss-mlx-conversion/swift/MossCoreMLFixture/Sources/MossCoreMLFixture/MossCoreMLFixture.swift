@preconcurrency import CoreML
import Darwin
import Foundation

struct Fixture: Decodable {
    let promptLen: Int
    let hiddenSize: Int
    let headDim: Int
    let ropeTheta: Double
    let inputIds: [Int32]?
    let audioInputMask: [Bool]?
    let promptPrefixIds: [Int32]?
    let promptSuffixIds: [Int32]?
    let audioTokenCount: Int?
    let audioPlaceholderId: Int32?
    let audioDataShape: [Int]
    let audioData: [Float]
    let audioDataSeqlens: [Int32]
    let generatedIds: [Int32]

    enum CodingKeys: String, CodingKey {
        case promptLen = "prompt_len"
        case hiddenSize = "hidden_size"
        case headDim = "head_dim"
        case ropeTheta = "rope_theta"
        case inputIds = "input_ids"
        case audioInputMask = "audio_input_mask"
        case promptPrefixIds = "prompt_prefix_ids"
        case promptSuffixIds = "prompt_suffix_ids"
        case audioTokenCount = "audio_token_count"
        case audioPlaceholderId = "audio_placeholder_id"
        case audioDataShape = "audio_data_shape"
        case audioData = "audio_data"
        case audioDataSeqlens = "audio_data_seqlens"
        case generatedIds = "generated_ids"
    }
}

struct Prompt {
    let inputIds: [Int32]
    let audioInputMask: [Bool]
    let source: String

    var promptLen: Int {
        inputIds.count
    }

    var audioTokenCount: Int {
        audioInputMask.filter { $0 }.count
    }
}

struct Options {
    var packagesDir = URL(fileURLWithPath: "coreml/build")
    var fixture = URL(fileURLWithPath: "artifacts/coreml/moss_swift_fixture.json")
    var output: URL?
    var tokenPackage = "compiled/moss_token_embedding.mlmodelc"
    var audioPackage = "compiled_audio/moss_audio_encoder_adapter_fixture.mlmodelc"
    var decoderPackage = "compiled_stateful/moss_decoder_stateful_fused.mlmodelc"
    var tokenizer = URL(fileURLWithPath: "artifacts/coreml/moss_tokenizer.json")
    var tokenMaxSeqLen = 512
    var maxNewTokens = 5
}

struct TopKEntry: Encodable {
    let index: Int
    let value: Float
}

struct Timing: Encodable {
    let tokenEmbeddingPrompt: Double
    let audioEncoderAdapter: Double
    let statefulDecoderPrefill: Double
    let tokenEmbeddingDecode: Double
    let statefulDecoderDecode: Double

    var total: Double {
        tokenEmbeddingPrompt + audioEncoderAdapter + statefulDecoderPrefill
            + tokenEmbeddingDecode + statefulDecoderDecode
    }

    enum CodingKeys: String, CodingKey {
        case tokenEmbeddingPrompt = "token_embedding_prompt"
        case audioEncoderAdapter = "audio_encoder_adapter"
        case statefulDecoderPrefill = "stateful_decoder_prefill"
        case tokenEmbeddingDecode = "token_embedding_decode"
        case statefulDecoderDecode = "stateful_decoder_decode"
        case total
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(tokenEmbeddingPrompt, forKey: .tokenEmbeddingPrompt)
        try container.encode(audioEncoderAdapter, forKey: .audioEncoderAdapter)
        try container.encode(statefulDecoderPrefill, forKey: .statefulDecoderPrefill)
        try container.encode(tokenEmbeddingDecode, forKey: .tokenEmbeddingDecode)
        try container.encode(statefulDecoderDecode, forKey: .statefulDecoderDecode)
        try container.encode(total, forKey: .total)
    }
}

struct Result: Encodable {
    let fixture: String
    let packagesDir: String
    let promptSource: String
    let promptLen: Int
    let audioTokenCount: Int
    let firstTokenId: Int
    let secondTokenId: Int
    let maxNewTokens: Int
    let generatedIds: [Int]
    let expectedGeneratedIds: [Int]
    let generatedPrefixMatchCount: Int
    let generatedPrefixMatchesExpected: Bool
    let generatedText: String
    let expectedText: String
    let normalizedGeneratedText: String
    let normalizedExpectedText: String
    let rawWer: Double
    let rawCer: Double
    let normalizedWer: Double
    let normalizedCer: Double
    let prefillTopK: [TopKEntry]
    let stepTopK: [TopKEntry]
    let prefillTop1MatchesFirstToken: Bool
    let stepTop1MatchesSecondToken: Bool
    let timingSeconds: Timing

    enum CodingKeys: String, CodingKey {
        case fixture
        case packagesDir = "packages_dir"
        case promptSource = "prompt_source"
        case promptLen = "prompt_len"
        case audioTokenCount = "audio_token_count"
        case firstTokenId = "first_token_id"
        case secondTokenId = "second_token_id"
        case maxNewTokens = "max_new_tokens"
        case generatedIds = "generated_ids"
        case expectedGeneratedIds = "expected_generated_ids"
        case generatedPrefixMatchCount = "generated_prefix_match_count"
        case generatedPrefixMatchesExpected = "generated_prefix_matches_expected"
        case generatedText = "generated_text"
        case expectedText = "expected_text"
        case normalizedGeneratedText = "normalized_generated_text"
        case normalizedExpectedText = "normalized_expected_text"
        case rawWer = "raw_wer"
        case rawCer = "raw_cer"
        case normalizedWer = "normalized_wer"
        case normalizedCer = "normalized_cer"
        case prefillTopK = "prefill_topk"
        case stepTopK = "step_topk"
        case prefillTop1MatchesFirstToken = "prefill_top1_matches_first_token"
        case stepTop1MatchesSecondToken = "step_top1_matches_second_token"
        case timingSeconds = "timing_seconds"
    }
}

enum RunnerError: Error, CustomStringConvertible {
    case missingArgument(String)
    case invalidArgument(String)
    case missingFeature(String)
    case unavailable(String)

    var description: String {
        switch self {
        case .missingArgument(let name):
            "missing value for \(name)"
        case .invalidArgument(let message):
            message
        case .missingFeature(let name):
            "CoreML output is missing feature \(name)"
        case .unavailable(let message):
            message
        }
    }
}

func parseOptions(_ arguments: [String]) throws -> Options {
    var options = Options()
    var index = 1
    while index < arguments.count {
        let argument = arguments[index]
        func value() throws -> String {
            guard index + 1 < arguments.count else {
                throw RunnerError.missingArgument(argument)
            }
            index += 1
            return arguments[index]
        }
        switch argument {
        case "--packages-dir":
            options.packagesDir = URL(fileURLWithPath: try value())
        case "--fixture":
            options.fixture = URL(fileURLWithPath: try value())
        case "--output":
            options.output = URL(fileURLWithPath: try value())
        case "--token-package":
            options.tokenPackage = try value()
        case "--audio-package":
            options.audioPackage = try value()
        case "--decoder-package":
            options.decoderPackage = try value()
        case "--tokenizer":
            options.tokenizer = URL(fileURLWithPath: try value())
        case "--token-max-seq-len":
            guard let parsed = Int(try value()) else {
                throw RunnerError.invalidArgument("invalid --token-max-seq-len")
            }
            options.tokenMaxSeqLen = parsed
        case "--max-new-tokens":
            guard let parsed = Int(try value()) else {
                throw RunnerError.invalidArgument("invalid --max-new-tokens")
            }
            options.maxNewTokens = parsed
        case "--help", "-h":
            print(
                """
                Usage: moss-coreml-fixture [--packages-dir DIR] [--fixture JSON] [--output JSON]
                                           [--token-package NAME] [--audio-package NAME]
                                           [--decoder-package NAME] [--token-max-seq-len N]
                                           [--max-new-tokens N] [--tokenizer JSON]
                """
            )
            Darwin.exit(0)
        default:
            throw RunnerError.invalidArgument("unknown argument \(argument)")
        }
        index += 1
    }
    return options
}

func loadFixture(_ url: URL) throws -> Fixture {
    let data = try Data(contentsOf: url)
    let decoder = JSONDecoder()
    return try decoder.decode(Fixture.self, from: data)
}

func resolvePrompt(_ fixture: Fixture) throws -> Prompt {
    if let prefixIds = fixture.promptPrefixIds,
       let suffixIds = fixture.promptSuffixIds,
       let audioTokenCount = fixture.audioTokenCount
    {
        guard audioTokenCount >= 0 else {
            throw RunnerError.invalidArgument("audio_token_count must be non-negative")
        }
        let placeholderId = fixture.audioPlaceholderId ?? 0
        let inputIds = prefixIds
            + [Int32](repeating: placeholderId, count: audioTokenCount)
            + suffixIds
        let audioInputMask = [Bool](repeating: false, count: prefixIds.count)
            + [Bool](repeating: true, count: audioTokenCount)
            + [Bool](repeating: false, count: suffixIds.count)
        guard inputIds.count == fixture.promptLen else {
            throw RunnerError.invalidArgument(
                "compact prompt length \(inputIds.count) != fixture prompt_len \(fixture.promptLen)"
            )
        }
        return Prompt(inputIds: inputIds, audioInputMask: audioInputMask, source: "compact")
    }

    guard let inputIds = fixture.inputIds, let audioInputMask = fixture.audioInputMask else {
        throw RunnerError.invalidArgument(
            "fixture needs either compact prompt fields or input_ids/audio_input_mask"
        )
    }
    guard inputIds.count == fixture.promptLen else {
        throw RunnerError.invalidArgument(
            "input_ids length \(inputIds.count) != fixture prompt_len \(fixture.promptLen)"
        )
    }
    guard audioInputMask.count == fixture.promptLen else {
        throw RunnerError.invalidArgument(
            "audio_input_mask length \(audioInputMask.count) != fixture prompt_len \(fixture.promptLen)"
        )
    }
    return Prompt(inputIds: inputIds, audioInputMask: audioInputMask, source: "serialized")
}

func makeFloatArray(shape: [Int], values: [Float]? = nil) throws -> MLMultiArray {
    let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
    if let values {
        guard values.count == array.count else {
            throw RunnerError.invalidArgument("value count \(values.count) != array count \(array.count)")
        }
        let pointer = array.dataPointer.bindMemory(to: Float.self, capacity: array.count)
        values.withUnsafeBufferPointer { source in
            pointer.update(from: source.baseAddress!, count: values.count)
        }
    }
    return array
}

func makeIntArray(shape: [Int], values: [Int32]) throws -> MLMultiArray {
    let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .int32)
    guard values.count == array.count else {
        throw RunnerError.invalidArgument("value count \(values.count) != array count \(array.count)")
    }
    let pointer = array.dataPointer.bindMemory(to: Int32.self, capacity: array.count)
    values.withUnsafeBufferPointer { source in
        pointer.update(from: source.baseAddress!, count: values.count)
    }
    return array
}

func paddedIds(_ ids: [Int32], maxSeqLen: Int) throws -> MLMultiArray {
    guard ids.count <= maxSeqLen else {
        throw RunnerError.invalidArgument("ids length \(ids.count) exceeds \(maxSeqLen)")
    }
    var padded = [Int32](repeating: 0, count: maxSeqLen)
    for (index, value) in ids.enumerated() {
        padded[index] = value
    }
    return try makeIntArray(shape: [1, maxSeqLen], values: padded)
}

func featureArray(_ provider: MLFeatureProvider, _ name: String) throws -> MLMultiArray {
    guard let array = provider.featureValue(for: name)?.multiArrayValue else {
        throw RunnerError.missingFeature(name)
    }
    return array
}

func strides(_ array: MLMultiArray) -> [Int] {
    array.strides.map(\.intValue)
}

func offset(_ indices: [Int], strides: [Int]) -> Int {
    zip(indices, strides).reduce(0) { partial, pair in
        partial + pair.0 * pair.1
    }
}

func floatValue(_ array: MLMultiArray, _ indices: [Int]) -> Float {
    let storageIndex = offset(indices, strides: strides(array))
    switch array.dataType {
    case .float32:
        return array.dataPointer.bindMemory(to: Float.self, capacity: array.count)[storageIndex]
    case .float16:
        return Float(array.dataPointer.bindMemory(to: Float16.self, capacity: array.count)[storageIndex])
    case .double:
        return Float(array.dataPointer.bindMemory(to: Double.self, capacity: array.count)[storageIndex])
    default:
        return array[indices.map { NSNumber(value: $0) }].floatValue
    }
}

func setFloat(_ array: MLMultiArray, _ indices: [Int], _ value: Float) {
    let storageIndex = offset(indices, strides: strides(array))
    array.dataPointer.bindMemory(to: Float.self, capacity: array.count)[storageIndex] = value
}

func buildMergedEmbeddings(
    tokenEmbeddings: MLMultiArray,
    audioEmbeddings: MLMultiArray,
    fixture: Fixture,
    prompt: Prompt
) throws -> MLMultiArray {
    let merged = try makeFloatArray(shape: [1, prompt.promptLen, fixture.hiddenSize])
    var audioIndex = 0
    for position in 0..<prompt.promptLen {
        let sourceIsAudio = prompt.audioInputMask[position]
        for hidden in 0..<fixture.hiddenSize {
            let value: Float
            if sourceIsAudio {
                value = floatValue(audioEmbeddings, [audioIndex, hidden])
            } else {
                value = floatValue(tokenEmbeddings, [0, position, hidden])
            }
            setFloat(merged, [0, position, hidden], value)
        }
        if sourceIsAudio {
            audioIndex += 1
        }
    }
    guard audioIndex == prompt.audioTokenCount else {
        throw RunnerError.invalidArgument(
            "merged \(audioIndex) audio embeddings but prompt expects \(prompt.audioTokenCount)"
        )
    }
    return merged
}

func buildRope(length: Int, start: Int, headDim: Int, ropeTheta: Double) throws -> (
    cos: MLMultiArray, sin: MLMultiArray
) {
    let cosArray = try makeFloatArray(shape: [1, length, headDim])
    let sinArray = try makeFloatArray(shape: [1, length, headDim])
    let half = headDim / 2
    for localPosition in 0..<length {
        let position = Double(start + localPosition)
        for index in 0..<half {
            let exponent = Double(2 * index) / Double(headDim)
            let angle = position / pow(ropeTheta, exponent)
            let cosine = Float(Darwin.cos(angle))
            let sine = Float(Darwin.sin(angle))
            setFloat(cosArray, [0, localPosition, index], cosine)
            setFloat(cosArray, [0, localPosition, index + half], cosine)
            setFloat(sinArray, [0, localPosition, index], sine)
            setFloat(sinArray, [0, localPosition, index + half], sine)
        }
    }
    return (cosArray, sinArray)
}

func buildCausalMask(length: Int) throws -> MLMultiArray {
    let mask = try makeFloatArray(shape: [1, 1, length, length])
    for query in 0..<length {
        for key in 0..<length {
            setFloat(mask, [0, 0, query, key], key > query ? -1_000_000_000.0 : 0.0)
        }
    }
    return mask
}

func topK(_ logits: MLMultiArray, count: Int = 5) -> [TopKEntry] {
    var winners: [TopKEntry] = []
    for index in 0..<logits.count {
        let value: Float
        switch logits.dataType {
        case .float32:
            value = logits.dataPointer.bindMemory(to: Float.self, capacity: logits.count)[index]
        case .float16:
            value = Float(logits.dataPointer.bindMemory(to: Float16.self, capacity: logits.count)[index])
        case .double:
            value = Float(logits.dataPointer.bindMemory(to: Double.self, capacity: logits.count)[index])
        default:
            value = logits[index].floatValue
        }
        winners.append(TopKEntry(index: index, value: value))
        winners.sort { $0.value > $1.value }
        if winners.count > count {
            winners.removeLast()
        }
    }
    return winners
}

func timedPrediction(
    model: MLModel,
    input: MLFeatureProvider,
    state: MLState? = nil
) async throws -> (MLFeatureProvider, Double) {
    let start = DispatchTime.now().uptimeNanoseconds
    let output: MLFeatureProvider
    if #available(macOS 15, *), let state {
        output = try await model.prediction(from: input, using: state)
    } else if state != nil {
        throw RunnerError.unavailable("MLState requires macOS 15+")
    } else {
        output = try await model.prediction(from: input)
    }
    let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000_000.0
    return (output, elapsed)
}

@main
struct MossCoreMLFixture {
    static func main() async throws {
        let options = try parseOptions(CommandLine.arguments)
        let fixture = try loadFixture(options.fixture)
        let prompt = try resolvePrompt(fixture)
        let maxNewTokens = min(options.maxNewTokens, fixture.generatedIds.count)
        guard maxNewTokens > 0 else {
            throw RunnerError.invalidArgument("--max-new-tokens must be positive")
        }
        let firstToken = Int(fixture.generatedIds[0])
        let secondToken = Int(fixture.generatedIds[1])
        let audioTokenCount = prompt.audioTokenCount
        let tokenizer = try QwenByteLevelTokenizer(tokenizerJSON: options.tokenizer)

        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        let tokenModel = try MLModel(
            contentsOf: options.packagesDir.appendingPathComponent(options.tokenPackage),
            configuration: configuration
        )
        let audioModel = try MLModel(
            contentsOf: options.packagesDir.appendingPathComponent(options.audioPackage),
            configuration: configuration
        )
        let decoderModel = try MLModel(
            contentsOf: options.packagesDir.appendingPathComponent(options.decoderPackage),
            configuration: configuration
        )
        guard #available(macOS 15, *) else {
            throw RunnerError.unavailable("stateful decoder requires macOS 15+")
        }
        let state = decoderModel.makeState()

        let tokenInput = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(
                multiArray: try paddedIds(prompt.inputIds, maxSeqLen: options.tokenMaxSeqLen)
            )
        ])
        let (tokenOutput, tokenSeconds) = try await timedPrediction(model: tokenModel, input: tokenInput)
        let tokenEmbeddings = try featureArray(tokenOutput, "token_embeddings")

        let audioInput = try MLDictionaryFeatureProvider(dictionary: [
            "audio_data": MLFeatureValue(
                multiArray: try makeFloatArray(shape: fixture.audioDataShape, values: fixture.audioData)
            ),
            "audio_data_seqlens": MLFeatureValue(
                multiArray: try makeIntArray(shape: [1], values: fixture.audioDataSeqlens)
            ),
        ])
        let (audioOutput, audioSeconds) = try await timedPrediction(model: audioModel, input: audioInput)
        let audioEmbeddings = try featureArray(audioOutput, "audio_embeddings")
        let mergedEmbeddings = try buildMergedEmbeddings(
            tokenEmbeddings: tokenEmbeddings,
            audioEmbeddings: audioEmbeddings,
            fixture: fixture,
            prompt: prompt
        )
        let (prefillCos, prefillSin) = try buildRope(
            length: prompt.promptLen,
            start: 0,
            headDim: fixture.headDim,
            ropeTheta: fixture.ropeTheta
        )
        let prefillMask = try buildCausalMask(length: prompt.promptLen)
        let prefillInput = try MLDictionaryFeatureProvider(dictionary: [
            "inputs_embeds": MLFeatureValue(multiArray: mergedEmbeddings),
            "cos": MLFeatureValue(multiArray: prefillCos),
            "sin": MLFeatureValue(multiArray: prefillSin),
            "attention_mask": MLFeatureValue(multiArray: prefillMask),
        ])
        let (prefillOutput, prefillSeconds) = try await timedPrediction(
            model: decoderModel,
            input: prefillInput,
            state: state
        )
        let prefillTopK = try topK(featureArray(prefillOutput, "logits"))
        guard let firstGenerated = prefillTopK.first?.index else {
            throw RunnerError.invalidArgument("prefill produced no logits")
        }

        var generatedIds = [firstGenerated]
        var firstStepTopK: [TopKEntry] = []
        var currentToken = Int32(firstGenerated)
        var decodeTokenSeconds = 0.0
        var decodeStepSeconds = 0.0
        for stepIndex in 0..<max(0, maxNewTokens - 1) {
            let tokenInput = try MLDictionaryFeatureProvider(dictionary: [
                "input_ids": MLFeatureValue(
                    multiArray: try paddedIds([currentToken], maxSeqLen: options.tokenMaxSeqLen)
                )
            ])
            let (tokenOutput, tokenDecodeSeconds) = try await timedPrediction(
                model: tokenModel,
                input: tokenInput
            )
            decodeTokenSeconds += tokenDecodeSeconds
            let tokenEmbeddingFull = try featureArray(tokenOutput, "token_embeddings")
            let tokenEmbedding = try makeFloatArray(shape: [1, 1, fixture.hiddenSize])
            for hidden in 0..<fixture.hiddenSize {
                setFloat(
                    tokenEmbedding,
                    [0, 0, hidden],
                    floatValue(tokenEmbeddingFull, [0, 0, hidden])
                )
            }

            let stepPosition = prompt.promptLen + stepIndex
            let (stepCos, stepSin) = try buildRope(
                length: 1,
                start: stepPosition,
                headDim: fixture.headDim,
                ropeTheta: fixture.ropeTheta
            )
            let stepMask = try makeFloatArray(shape: [1, 1, 1, stepPosition + 1])
            let stepInput = try MLDictionaryFeatureProvider(dictionary: [
                "inputs_embeds": MLFeatureValue(multiArray: tokenEmbedding),
                "cos": MLFeatureValue(multiArray: stepCos),
                "sin": MLFeatureValue(multiArray: stepSin),
                "attention_mask": MLFeatureValue(multiArray: stepMask),
            ])
            let (stepOutput, stepSeconds) = try await timedPrediction(
                model: decoderModel,
                input: stepInput,
                state: state
            )
            decodeStepSeconds += stepSeconds
            let stepTopK = try topK(featureArray(stepOutput, "logits"))
            if stepIndex == 0 {
                firstStepTopK = stepTopK
            }
            guard let nextToken = stepTopK.first?.index else {
                throw RunnerError.invalidArgument("decode step produced no logits")
            }
            generatedIds.append(nextToken)
            currentToken = Int32(nextToken)
        }
        let expectedGeneratedIds = fixture.generatedIds.prefix(maxNewTokens).map { Int($0) }
        var generatedPrefixMatchCount = 0
        for (actual, expected) in zip(generatedIds, expectedGeneratedIds) {
            guard actual == expected else { break }
            generatedPrefixMatchCount += 1
        }
        let generatedPrefixMatchesExpected = generatedPrefixMatchCount == expectedGeneratedIds.count
        let generatedText = tokenizer.decode(generatedIds)
        let expectedText = tokenizer.decode(expectedGeneratedIds)
        let normalizedGeneratedText = normalizedTranscript(generatedText)
        let normalizedExpectedText = normalizedTranscript(expectedText)

        let result = Result(
            fixture: options.fixture.path,
            packagesDir: options.packagesDir.path,
            promptSource: prompt.source,
            promptLen: prompt.promptLen,
            audioTokenCount: audioTokenCount,
            firstTokenId: firstToken,
            secondTokenId: secondToken,
            maxNewTokens: maxNewTokens,
            generatedIds: generatedIds,
            expectedGeneratedIds: expectedGeneratedIds,
            generatedPrefixMatchCount: generatedPrefixMatchCount,
            generatedPrefixMatchesExpected: generatedPrefixMatchesExpected,
            generatedText: generatedText,
            expectedText: expectedText,
            normalizedGeneratedText: normalizedGeneratedText,
            normalizedExpectedText: normalizedExpectedText,
            rawWer: wordErrorRate(reference: expectedText, hypothesis: generatedText),
            rawCer: characterErrorRate(reference: expectedText, hypothesis: generatedText),
            normalizedWer: wordErrorRate(
                reference: normalizedExpectedText,
                hypothesis: normalizedGeneratedText
            ),
            normalizedCer: characterErrorRate(
                reference: normalizedExpectedText,
                hypothesis: normalizedGeneratedText
            ),
            prefillTopK: prefillTopK,
            stepTopK: firstStepTopK,
            prefillTop1MatchesFirstToken: prefillTopK.first?.index == firstToken,
            stepTop1MatchesSecondToken: generatedIds.count > 1 && generatedIds[1] == secondToken,
            timingSeconds: Timing(
                tokenEmbeddingPrompt: tokenSeconds,
                audioEncoderAdapter: audioSeconds,
                statefulDecoderPrefill: prefillSeconds,
                tokenEmbeddingDecode: decodeTokenSeconds,
                statefulDecoderDecode: decodeStepSeconds
            )
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(result)
        if let output = options.output {
            try FileManager.default.createDirectory(
                at: output.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try data.write(to: output)
        }
        print(String(decoding: data, as: UTF8.self))
    }
}
