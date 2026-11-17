import Foundation
import AppKit
import CoreImage
import CryptoKit

@main struct StudioBackgroundPreviewSmoke {
    static func bytes(_ image: CGImage) -> [UInt8] {
        var result = [UInt8](repeating: 0, count: image.width * image.height * 4)
        result.withUnsafeMutableBytes { data in
            let context = CGContext(data: data.baseAddress, width: image.width, height: image.height,
                bitsPerComponent: 8, bytesPerRow: image.width * 4, space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
            context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        }
        return result
    }
    @MainActor static func main() async throws {
        let source = URL(fileURLWithPath: CommandLine.arguments[1])
        let before = SHA256.hash(data: try Data(contentsOf: source))
        let temporary = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: temporary) }
        let renderer = StudioPhotoPreviewRenderer()
        let originalRequest = StudioPhotoPreviewRequest(photo: source.path, backgroundMode: "Original", backgroundPath: "", color: .red)
        guard let original = try await renderer.render(originalRequest) else { fatalError("No source preview") }
        precondition(max(original.width, original.height) <= 1600)
        var request = originalRequest; request.backgroundMode = "Solid color"
        guard let red = try await renderer.render(request) else { fatalError("No solid-color preview") }
        let sourcePixels = bytes(original), redPixels = bytes(red)
        let corner = (5 * red.width + 5) * 4
        precondition(redPixels[corner] > 245 && redPixels[corner+1] < 10 && redPixels[corner+2] < 10)
        // An interior subject pixel retains real clothing/skin, rather than
        // receiving the solid fill intended for the background.
        let person = (red.height / 2 * red.width + red.width / 2) * 4
        for channel in 0..<3 { precondition(abs(Int(redPixels[person+channel])-Int(sourcePixels[person+channel])) < 5) }
        let exported = temporary.appendingPathComponent("export-preparation.png")
        try PhotoPreparation.composite(photo: source, background: nil, color: .red, output: exported)
        let exportPixels = bytes(PhotoPreview.load(path: exported.path)!)
        precondition(exportPixels.count == redPixels.count)
        let maximumDifference = zip(exportPixels, redPixels).map { abs(Int($0)-Int($1)) }.max()!
        precondition(maximumDifference <= 3, "Preview differs from shared export preparation: \(maximumDifference)")
        let background = temporary.appendingPathComponent("background.png")
        try CIContext().writePNGRepresentation(of: CIImage(color: CIColor(red: 0, green: 0, blue: 1)).cropped(to: CGRect(x: 0, y: 0, width: 64, height: 64)),
            to: background, format: .RGBA8, colorSpace: CGColorSpaceCreateDeviceRGB())
        request.backgroundMode = "Image"; request.backgroundPath = background.path
        let blue = bytes(try await renderer.render(request)!)
        precondition(blue[corner] < 10 && blue[corner+2] > 245)
        request.backgroundPath = "/nonexistent/background.png"
        do { _ = try await renderer.render(request); fatalError("Invalid background accepted") }
        catch { precondition(error.localizedDescription.contains("background image")) }
        let large = temporary.appendingPathComponent("large-real-photo.png")
        let enlarged = CIImage(contentsOf: source)!.transformed(by: CGAffineTransform(scaleX: 2, y: 2))
        try CIContext().writePNGRepresentation(of: enlarged, to: large, format: .RGBA8, colorSpace: CGColorSpaceCreateDeviceRGB())
        var largeRequest = originalRequest; largeRequest.photo = large.path; largeRequest.backgroundMode = "Solid color"
        let bounded = try await renderer.render(largeRequest)!
        precondition(max(bounded.width, bounded.height) <= 1600)
        let cancelled = Task { try await renderer.render(originalRequest) }
        cancelled.cancel()
        do { _ = try await cancelled.value; fatalError("Cancelled preview processed") }
        catch is CancellationError { }
        let after = SHA256.hash(data: try Data(contentsOf: source))
        precondition(before == after)
        print("Real-photo original/color/image previews, matching export pixels, missing-background errors, cancellation and unchanged source: passed")
    }
}
