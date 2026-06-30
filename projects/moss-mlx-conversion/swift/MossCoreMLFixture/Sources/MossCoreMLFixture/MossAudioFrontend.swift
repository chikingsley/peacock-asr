@preconcurrency import AVFoundation
import Accelerate
import Darwin
import Foundation
import os

struct MossAudioFeatures {
    let source: String
    let shape: [Int]
    let data: [Float]
    let seqlens: [Int32]
}

struct AudioFeatureDiff: Encodable {
    let comparedValues: Int
    let maxAbs: Float
    let meanAbs: Float

    enum CodingKeys: String, CodingKey {
        case comparedValues = "compared_values"
        case maxAbs = "max_abs"
        case meanAbs = "mean_abs"
    }
}

func compareAudioFeatures(_ left: [Float], _ right: [Float]) -> AudioFeatureDiff? {
    guard left.count == right.count, !left.isEmpty else {
        return nil
    }
    var maxAbs: Float = 0
    var sum: Float = 0
    for (leftValue, rightValue) in zip(left, right) {
        let diff = abs(leftValue - rightValue)
        maxAbs = max(maxAbs, diff)
        sum += diff
    }
    return AudioFeatureDiff(
        comparedValues: left.count,
        maxAbs: maxAbs,
        meanAbs: sum / Float(left.count)
    )
}

func mossAudioTokenCount(melFrames: Int) -> Int {
    func floorDiv(_ value: Int, _ divisor: Int) -> Int {
        if value >= 0 {
            return value / divisor
        }
        return -((-value + divisor - 1) / divisor)
    }

    let remainingFrames = melFrames % 100
    let featureLength = floorDiv(remainingFrames - 1, 2) + 1
    let localLength = floorDiv(floorDiv(featureLength - 1, 2), 2) + 1
    return localLength + (melFrames / 100) * 13
}

enum MossAudioFile {
    static func loadMono16k(url: URL) throws -> [Float] {
        let file = try AVAudioFile(forReading: url)
        let inputFormat = file.processingFormat
        guard let inputBuffer = AVAudioPCMBuffer(
            pcmFormat: inputFormat,
            frameCapacity: AVAudioFrameCount(file.length)
        ) else {
            throw RunnerError.invalidArgument("could not allocate input audio buffer")
        }
        try file.read(into: inputBuffer)

        guard let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: 16_000,
            channels: 1,
            interleaved: false
        ) else {
            throw RunnerError.invalidArgument("could not create 16 kHz mono target format")
        }

        if inputFormat.sampleRate == targetFormat.sampleRate,
           inputFormat.channelCount == targetFormat.channelCount,
           inputFormat.commonFormat == targetFormat.commonFormat,
           inputFormat.isInterleaved == targetFormat.isInterleaved
        {
            return extractMonoFloat32(inputBuffer)
        }

        guard let converter = AVAudioConverter(from: inputFormat, to: targetFormat) else {
            throw RunnerError.invalidArgument("could not create audio converter")
        }
        converter.sampleRateConverterAlgorithm = AVSampleRateConverterAlgorithm_Mastering
        converter.sampleRateConverterQuality = AVAudioQuality.max.rawValue

        let ratio = targetFormat.sampleRate / inputFormat.sampleRate
        let outputCapacity = AVAudioFrameCount(
            (Double(inputBuffer.frameLength) * ratio).rounded(.up) + 4096
        )
        guard let outputBuffer = AVAudioPCMBuffer(
            pcmFormat: targetFormat,
            frameCapacity: outputCapacity
        ) else {
            throw RunnerError.invalidArgument("could not allocate converted audio buffer")
        }

        let provided = OSAllocatedUnfairLock(initialState: false)
        let inputBlock: AVAudioConverterInputBlock = { _, status in
            let alreadyProvided = provided.withLock { state -> Bool in
                if state {
                    return true
                }
                state = true
                return false
            }
            if alreadyProvided {
                status.pointee = .endOfStream
                return nil
            }
            status.pointee = .haveData
            return inputBuffer
        }

        var error: NSError?
        let status = converter.convert(to: outputBuffer, error: &error, withInputFrom: inputBlock)
        if status == .error {
            throw RunnerError.invalidArgument(
                "audio conversion failed: \(error?.localizedDescription ?? "unknown error")"
            )
        }
        return extractMonoFloat32(outputBuffer)
    }

    private static func extractMonoFloat32(_ buffer: AVAudioPCMBuffer) -> [Float] {
        guard let channelData = buffer.floatChannelData else {
            return []
        }
        return Array(
            UnsafeBufferPointer(start: channelData[0], count: Int(buffer.frameLength))
        )
    }
}

final class WhisperLogMelFrontend {
    private let sampleRate = 16_000
    private let nFFT = 400
    private let hopLength = 160
    private let nMels = 128
    private let melFloor: Float = 1e-10
    private let hannWindow: [Float]
    private let melFilters: [[Float]]
    private let dftCosTable: [Float]
    private let dftSinTable: [Float]

    init() throws {
        self.hannWindow = Self.periodicHannWindow(length: nFFT)
        self.melFilters = Self.slaneyMelFilter(
            nFFT: nFFT,
            nMels: nMels,
            sampleRate: sampleRate,
            fMin: 0,
            fMax: 8_000
        )
        let tables = Self.makeDFTTables(nFFT: nFFT)
        self.dftCosTable = tables.cos
        self.dftSinTable = tables.sin
    }

    func compute(samples: [Float], source: String) throws -> MossAudioFeatures {
        guard samples.count > nFFT else {
            throw RunnerError.invalidArgument("audio is too short for \(nFFT)-sample FFT")
        }
        let padded = try reflectPadded(samples, padding: nFFT / 2)
        let fullFrameCount = 1 + (padded.count - nFFT) / hopLength
        let frameCount = fullFrameCount - 1
        guard frameCount > 0 else {
            throw RunnerError.invalidArgument("audio produced no mel frames")
        }

        var logMels = [Float](repeating: 0, count: nMels * frameCount)
        var realIn = [Float](repeating: 0, count: nFFT)
        var power = [Float](repeating: 0, count: nFFT / 2 + 1)

        for frameIndex in 0..<frameCount {
            let start = frameIndex * hopLength
            for index in 0..<nFFT {
                realIn[index] = padded[start + index] * hannWindow[index]
            }

            for bin in 0..<power.count {
                var real: Float = 0
                var imaginary: Float = 0
                let tableOffset = bin * nFFT
                for index in 0..<nFFT {
                    let value = realIn[index]
                    real += value * dftCosTable[tableOffset + index]
                    imaginary -= value * dftSinTable[tableOffset + index]
                }
                power[bin] = real * real + imaginary * imaginary
            }

            for melIndex in 0..<nMels {
                let filter = melFilters[melIndex]
                var melEnergy: Float = 0
                for bin in 0..<power.count {
                    melEnergy += filter[bin] * power[bin]
                }
                let value = log10(max(melFloor, melEnergy))
                logMels[melIndex * frameCount + frameIndex] = value
            }
        }

        guard let maxLogMel = logMels.max() else {
            throw RunnerError.invalidArgument("mel computation produced no values")
        }
        let floor = maxLogMel - 8.0
        for index in logMels.indices {
            logMels[index] = (max(logMels[index], floor) + 4.0) / 4.0
        }

        return MossAudioFeatures(
            source: source,
            shape: [nMels, frameCount],
            data: logMels,
            seqlens: [Int32(frameCount)]
        )
    }

    private func reflectPadded(_ samples: [Float], padding: Int) throws -> [Float] {
        guard samples.count > padding else {
            throw RunnerError.invalidArgument("audio is too short for reflect padding")
        }
        var padded = [Float](repeating: 0, count: samples.count + 2 * padding)
        for index in 0..<padding {
            padded[index] = samples[padding - index]
        }
        for index in samples.indices {
            padded[padding + index] = samples[index]
        }
        for index in 0..<padding {
            padded[padding + samples.count + index] = samples[samples.count - 2 - index]
        }
        return padded
    }

    private static func periodicHannWindow(length: Int) -> [Float] {
        (0..<length).map { index in
            0.5 * (1.0 - cos(2.0 * Float.pi * Float(index) / Float(length)))
        }
    }

    private static func makeDFTTables(nFFT: Int) -> (cos: [Float], sin: [Float]) {
        let bins = nFFT / 2 + 1
        var cosTable = [Float](repeating: 0, count: bins * nFFT)
        var sinTable = [Float](repeating: 0, count: bins * nFFT)
        for bin in 0..<bins {
            let tableOffset = bin * nFFT
            for index in 0..<nFFT {
                let angle = 2.0 * Float.pi * Float(bin * index) / Float(nFFT)
                cosTable[tableOffset + index] = cos(angle)
                sinTable[tableOffset + index] = sin(angle)
            }
        }
        return (cosTable, sinTable)
    }

    private static func slaneyMelFilter(
        nFFT: Int,
        nMels: Int,
        sampleRate: Int,
        fMin: Float,
        fMax: Float
    ) -> [[Float]] {
        let nBins = nFFT / 2 + 1
        let melMin = hzToMelSlaney(fMin)
        let melMax = hzToMelSlaney(fMax)
        let melStep = (melMax - melMin) / Float(nMels + 1)
        let melPoints = (0..<(nMels + 2)).map { melMin + Float($0) * melStep }
        let hzPoints = melPoints.map(melToHzSlaney)
        let fftFrequencies = (0..<nBins).map {
            Float($0) * Float(sampleRate) / Float(nFFT)
        }

        var filters = [[Float]](
            repeating: [Float](repeating: 0, count: nBins),
            count: nMels
        )
        for melIndex in 0..<nMels {
            let lower = hzPoints[melIndex]
            let center = hzPoints[melIndex + 1]
            let upper = hzPoints[melIndex + 2]
            let leftDenominator = max(center - lower, 1e-10)
            let rightDenominator = max(upper - center, 1e-10)
            let enorm = 2.0 / max(upper - lower, 1e-10)
            for (bin, frequency) in fftFrequencies.enumerated() {
                let weight: Float
                if frequency < lower || frequency > upper {
                    weight = 0
                } else if frequency <= center {
                    weight = (frequency - lower) / leftDenominator
                } else {
                    weight = (upper - frequency) / rightDenominator
                }
                filters[melIndex][bin] = weight * enorm
            }
        }
        return filters
    }

    private static func hzToMelSlaney(_ hz: Float) -> Float {
        let linearScale: Float = 200.0 / 3.0
        let minLogHz: Float = 1_000.0
        let minLogMel = minLogHz / linearScale
        let logStep = Darwin.logf(6.4) / 27.0
        if hz >= minLogHz {
            return minLogMel + Darwin.logf(hz / minLogHz) / logStep
        }
        return hz / linearScale
    }

    private static func melToHzSlaney(_ mel: Float) -> Float {
        let linearScale: Float = 200.0 / 3.0
        let minLogHz: Float = 1_000.0
        let minLogMel = minLogHz / linearScale
        let logStep = Darwin.logf(6.4) / 27.0
        if mel >= minLogMel {
            return minLogHz * Darwin.expf(logStep * (mel - minLogMel))
        }
        return linearScale * mel
    }
}
