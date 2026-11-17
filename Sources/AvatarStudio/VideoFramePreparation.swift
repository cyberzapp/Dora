import Foundation
import AVFoundation
import ImageIO

struct VideoFramePreparation {
    static func prepare(source: URL, seconds: Double, directory: URL) async throws -> Outfit {
        let asset = AVURLAsset(url: source)
        let duration = try await asset.load(.duration).seconds
        guard duration.isFinite, duration > 0, seconds.isFinite, seconds >= 0, seconds < duration else {
            throw StudioError.message("Choose a frame inside this video.")
        }
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero
        let time = CMTime(seconds: seconds, preferredTimescale: 60000)
        let (image, actualTime) = try await generator.image(at: time)
        try Task.checkCancellation()
        let id = UUID().uuidString
        let video = directory.appendingPathComponent("\(id)-source.\(source.pathExtension)")
        let frame = directory.appendingPathComponent("\(id)-frame.png")
        var copied = false
        var prepared: Outfit?
        defer { try? FileManager.default.removeItem(at: frame) }
        do {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            // Keep the complete source recording as an unchanged local copy.
            try FileManager.default.copyItem(at: source, to: video); copied = true
            guard let destination = CGImageDestinationCreateWithURL(frame as CFURL, "public.png" as CFString, 1, nil) else {
                throw StudioError.message("Cannot save this video frame")
            }
            CGImageDestinationAddImage(destination, image, nil)
            guard CGImageDestinationFinalize(destination) else { throw StudioError.message("Cannot save this video frame") }
            try Task.checkCancellation()
            var outfit = try PhotoPreparation.prepare(source: frame, directory: directory)
            prepared = outfit
            outfit.name = source.deletingPathExtension().lastPathComponent + " • video frame"
            outfit.sourceVideo = video.path; outfit.sourceFrameSeconds = actualTime.seconds
            outfit.guidance += " Appearance extracted from a recorded frame at \(String(format: "%.2f", actualTime.seconds)) s. Video motion is not transferred by photo mode."
            try Task.checkCancellation()
            return outfit
        } catch {
            if copied { try? FileManager.default.removeItem(at: video) }
            if let outfit = prepared {
                for path in [outfit.original, outfit.prepared, directory.appendingPathComponent("\(outfit.id)-landmarks.json").path] {
                    try? FileManager.default.removeItem(atPath: path)
                }
            }
            throw error
        }
    }
}
