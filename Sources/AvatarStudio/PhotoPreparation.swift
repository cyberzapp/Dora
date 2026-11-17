import Foundation
import AppKit
import Vision
import CoreImage
import ImageIO

struct PhotoPreparation {
    static func renderSource(source: URL, output: URL) throws {
        guard let input = CIImage(contentsOf: source, options: [.applyOrientationProperty: true]),
              let image = CIContext().createCGImage(input, from: input.extent) else {
            throw StudioError.message("The original outfit photo is missing or cannot be decoded. Import it again.")
        }
        let temporary = output.appendingPathExtension("partial")
        defer { try? FileManager.default.removeItem(at: temporary) }
        guard let destination = CGImageDestinationCreateWithURL(temporary as CFURL, "public.png" as CFString, 1, nil) else {
            throw StudioError.message("Cannot prepare the full-resolution outfit photo")
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { throw StudioError.message("Cannot save the full-resolution outfit photo") }
        try FileManager.default.moveItem(at: temporary, to: output)
    }
    static func prepare(source: URL, directory: URL) throws -> Outfit {
        let id = UUID().uuidString
        let original = directory.appendingPathComponent("\(id).\(source.pathExtension)")
        let prepared = directory.appendingPathComponent("\(id)-prepared.png")
        let metadata = directory.appendingPathComponent("\(id)-landmarks.json")
        guard let input = CIImage(contentsOf: source, options: [.applyOrientationProperty: true]) else { throw StudioError.message("This image could not be decoded. Import a JPEG, PNG, or HEIC photo.") }
        let size = input.extent.size
        guard min(size.width, size.height) >= 512 else { throw StudioError.message("Use a photo at least 512 pixels on each side, with a clearly visible face.") }
        let context = CIContext()
        let scale = min(1, 1280 / max(size.width, size.height))
        let image = input.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        guard let cg = context.createCGImage(image, from: image.extent) else { throw StudioError.message("Unable to prepare photo") }
        let request = VNDetectFaceLandmarksRequest()
        try VNImageRequestHandler(cgImage: cg).perform([request])
        guard let faces = request.results, faces.count == 1, let face = faces.first else { throw StudioError.message("Use a photo with exactly one unobstructed face looking toward the camera.") }
        guard face.boundingBox.width * CGFloat(cg.width) >= 100, face.landmarks?.outerLips != nil else { throw StudioError.message("Your face is too small or obscured. Choose a closer, front-facing photograph.") }
        let b = face.boundingBox
        guard b.minX > 0.01, b.maxX < 0.99, b.minY > 0.01, b.maxY < 0.99 else { throw StudioError.message("The face touches the image edge. Upload a photo with space around your head.") }
        let quality = VNDetectFaceCaptureQualityRequest()
        try VNImageRequestHandler(cgImage: cg).perform([quality])
        let score = Double(quality.results?.first?.faceCaptureQuality ?? 0)
        let blur = blurVariance(cg)
        var guidance = "Face detected. Source composition preserved."
        if blur < 25 { guidance += " Image detail appears soft; use a sharper photo if the rendered face looks blurred." }
        if score < 0.35 { guidance += " Face quality is low: use sharper focus, even lighting, and a forward-facing pose." }
        if b.height > 0.5 { guidance += " For a seated presenter, upload a wider photo showing your shoulders and torso." }
        try FileManager.default.copyItem(at: source, to: original)
        do {
            // Face analysis is bounded for responsiveness. Keep all available
            // source detail in the separate orientation-normalized render cache.
            guard let fullResolution = context.createCGImage(input, from: input.extent) else { throw StudioError.message("Unable to preserve the full-resolution photograph") }
            guard let destination = CGImageDestinationCreateWithURL(prepared as CFURL, "public.png" as CFString, 1, nil) else { throw StudioError.message("Cannot create prepared photo") }
            CGImageDestinationAddImage(destination, fullResolution, nil)
            guard CGImageDestinationFinalize(destination) else { throw StudioError.message("Cannot write prepared photo") }
            let landmarks = face.landmarks?.allPoints?.normalizedPoints.map { [Double($0.x), Double($0.y)] } ?? []
            try JSONSerialization.data(withJSONObject: ["version": 2, "face": [b.minX, b.minY, b.width, b.height], "landmarks": landmarks, "quality": score, "prepared_size": [fullResolution.width, fullResolution.height], "analysis_size": [cg.width, cg.height]]).write(to: metadata, options: .atomic)
        } catch {
            for path in [original, prepared, metadata] { try? FileManager.default.removeItem(at: path) }
            throw error
        }
        return Outfit(id: id, name: source.deletingPathExtension().lastPathComponent, original: original.path, prepared: prepared.path, guidance: guidance)
    }
    static func blurVariance(_ image: CGImage) -> Double {
        let n = 256
        var pixels = [UInt8](repeating: 0, count: n*n)
        return pixels.withUnsafeMutableBytes { raw in
            guard let context = CGContext(data: raw.baseAddress, width: n, height: n, bitsPerComponent: 8, bytesPerRow: n, space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGImageAlphaInfo.none.rawValue) else { return 0 }
            context.draw(image, in: CGRect(x: 0, y: 0, width: n, height: n))
            let bytes = raw.bindMemory(to: UInt8.self)
            var sum = 0.0, squares = 0.0, count = 0.0
            for y in 1..<(n-1) { for x in 1..<(n-1) {
                let i = y*n+x
                let value = Double(Int(bytes[i-1])+Int(bytes[i+1])+Int(bytes[i-n])+Int(bytes[i+n])-4*Int(bytes[i]))
                sum += value; squares += value*value; count += 1
            } }
            return squares/count-pow(sum/count, 2)
        }
    }
    static func composite(photo: URL, background: URL?, color: NSColor, output: URL) throws {
        guard let input = CIImage(contentsOf: photo) else { throw StudioError.message("Photo unavailable") }
        let result = try compositeImage(input: input, background: background, color: color)
        try CIContext().writePNGRepresentation(of: result, to: output, format: .RGBA8, colorSpace: CGColorSpaceCreateDeviceRGB())
    }
    /// Shared by export preparation and the native source preview. It changes
    /// only the background; original photographs remain untouched.
    static func compositeImage(input: CIImage, background: URL?, color: NSColor) throws -> CIImage {
        let backgroundImage: CIImage?
        if let background {
            guard let decoded = CIImage(contentsOf: background, options: [.applyOrientationProperty: true]),
                  !decoded.extent.isEmpty, !decoded.extent.isInfinite else {
                throw StudioError.message("The background image is missing or cannot be decoded. Import a valid background image again.")
            }
            backgroundImage = decoded
        } else { backgroundImage = nil }
        guard let fill = CIColor(color: color) else { throw StudioError.message("Choose an RGB background color.") }
        let request = VNGeneratePersonSegmentationRequest()
        request.qualityLevel = .accurate
        request.outputPixelFormat = kCVPixelFormatType_OneComponent8
        try VNImageRequestHandler(ciImage: input).perform([request])
        guard let buffer = request.results?.first?.pixelBuffer else { throw StudioError.message("Unable to separate the person from the background") }
        let mask = CIImage(cvPixelBuffer: buffer)
        let scaledMask = mask.transformed(by: CGAffineTransform(scaleX: input.extent.width / mask.extent.width, y: input.extent.height / mask.extent.height))
        var back = CIImage(color: fill).cropped(to: input.extent)
        if let bg = backgroundImage {
            let scale = max(input.extent.width / bg.extent.width, input.extent.height / bg.extent.height)
            let scaled = bg.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
            back = scaled.transformed(by: CGAffineTransform(translationX: (input.extent.width-scaled.extent.width)/2, y: (input.extent.height-scaled.extent.height)/2)).cropped(to: input.extent)
        }
        return input.applyingFilter("CIBlendWithMask", parameters: [kCIInputBackgroundImageKey: back, kCIInputMaskImageKey: scaledMask])
    }
}
