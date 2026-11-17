import Foundation
import CryptoKit
import CoreImage

@main struct VideoFrameImportSmoke {
    static func main() async throws {
        let source = URL(fileURLWithPath: CommandLine.arguments[1])
        let directory = URL(fileURLWithPath: CommandLine.arguments[2])
        let original = SHA256.hash(data: try Data(contentsOf: source))
        let outfit = try await VideoFramePreparation.prepare(source: source, seconds: 1.125, directory: directory)
        guard let savedVideo = outfit.sourceVideo, let actual = outfit.sourceFrameSeconds else { fatalError("Source video provenance missing") }
        let savedHash = SHA256.hash(data: try Data(contentsOf: URL(fileURLWithPath: savedVideo)))
        precondition(original == savedHash, "Source recording was changed")
        precondition(abs(actual - 1.125) < 1.0 / 24)
        let image = CIImage(contentsOf: URL(fileURLWithPath: outfit.prepared))!
        precondition(image.extent.size == CGSize(width: 1920, height: 1080), "Video frame lost resolution")
        let roundtrip = try JSONDecoder().decode(Outfit.self, from: JSONEncoder().encode(outfit))
        precondition(roundtrip.sourceVideo == savedVideo && roundtrip.sourceFrameSeconds == actual)
        let oldPaths = Set(try FileManager.default.contentsOfDirectory(atPath: directory.path))
        do {
            _ = try await VideoFramePreparation.prepare(source: source, seconds: 100000, directory: directory)
            fatalError("Out-of-range frame was accepted")
        } catch { }
        let newPaths = Set(try FileManager.default.contentsOfDirectory(atPath: directory.path))
        precondition(oldPaths == newPaths, "Invalid frame left orphan files")
        let finalHash = SHA256.hash(data: try Data(contentsOf: source))
        precondition(original == finalHash)
        print("Actual video-frame import, full-resolution preparation, unchanged source and invalid-time recovery: passed")
    }
}
