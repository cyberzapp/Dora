import Foundation
import AVFoundation

struct PreparedAudio: Sendable {
    let path: String
    let waveform: [Float]
    let duration: Double
}

struct AudioPreparation {
    /// Save an unprocessed prefix at the source sample rate and channel count.
    /// The imported recording remains the full-render source.
    static func writePreview(source: URL, output: URL, maximumSeconds: Double) throws {
        let input = try AVAudioFile(forReading: source, commonFormat: .pcmFormatFloat32, interleaved: false)
        let format = input.processingFormat
        let frames = min(input.length, AVAudioFramePosition((maximumSeconds * format.sampleRate).rounded(.down)))
        guard frames >= AVAudioFramePosition((0.5 * format.sampleRate).rounded(.up)),
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 8192) else {
            throw StudioError.message("The recording is too short to create a test clip.")
        }
        let settings: [String: Any] = [AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: format.sampleRate, AVNumberOfChannelsKey: format.channelCount,
            AVLinearPCMBitDepthKey: 32, AVLinearPCMIsFloatKey: true,
            AVLinearPCMIsBigEndianKey: false, AVLinearPCMIsNonInterleaved: false]
        let writer = try AVAudioFile(forWriting: output, settings: settings, commonFormat: .pcmFormatFloat32, interleaved: false)
        var remaining = frames
        while remaining > 0 {
            try input.read(into: buffer, frameCount: AVAudioFrameCount(min(remaining, 8192)))
            guard buffer.frameLength > 0 else { throw StudioError.message("The recording ended while preparing the test clip.") }
            try writer.write(from: buffer)
            remaining -= AVAudioFramePosition(buffer.frameLength)
        }
    }

    static func prepare(source: URL, directory: URL) throws -> PreparedAudio {
        let target = directory.appendingPathComponent("\(UUID().uuidString).\(source.pathExtension)")
        var copied = false
        do {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            try FileManager.default.copyItem(at: source, to: target); copied = true
            let file: AVAudioFile
            do { file = try AVAudioFile(forReading: target, commonFormat: .pcmFormatFloat32, interleaved: false) }
            catch { throw StudioError.message("This recording could not be decoded. Try a WAV, AIFF, MP3 or M4A file. Your previous voice-over has been kept.") }
            let duration = Double(file.length) / file.processingFormat.sampleRate
            guard duration.isFinite, duration >= 0.5, duration <= 300 else { throw StudioError.message("Use 0.5–300 seconds of audio. Your previous voice-over has been kept.") }
            let count = AVAudioFrameCount(max(1, file.length / 180))
            guard let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: count) else { throw StudioError.message("Cannot decode this recording. Try exporting it as a WAV file.") }
            var values: [Float] = []
            while file.framePosition < file.length {
                try file.read(into: buffer, frameCount: count)
                guard let channels = buffer.floatChannelData, buffer.frameLength > 0 else { throw StudioError.message("The recording ended unexpectedly. Import a complete audio file.") }
                var peak: Float = 0
                for channel in 0..<Int(buffer.format.channelCount) {
                    for index in 0..<Int(buffer.frameLength) {
                        let sample = channels[channel][index]
                        guard sample.isFinite else { throw StudioError.message("This recording contains invalid audio samples. Export a clean WAV file and import it again. Your previous voice-over has been kept.") }
                        peak = max(peak, abs(sample))
                    }
                }
                values.append(min(1, peak)) // Bound display amplitude; audio bytes remain untouched.
            }
            guard !values.isEmpty else { throw StudioError.message("The recording contains no readable audio.") }
            return PreparedAudio(path: target.path, waveform: values, duration: duration)
        } catch {
            if copied { try? FileManager.default.removeItem(at: target) }
            throw error
        }
    }
}
