import SwiftUI
import CoreImage

private typealias PreviewViewState<Value> = SwiftUI.State<Value>

struct StudioPhotoPreviewRequest: Hashable, Sendable {
    var photo: String
    var backgroundMode: String
    var backgroundPath: String
    var rgba: [Double]

    init(photo: String, backgroundMode: String, backgroundPath: String, color: NSColor) {
        self.photo = photo; self.backgroundMode = backgroundMode; self.backgroundPath = backgroundPath
        let rgb = color.usingColorSpace(.sRGB) ?? .darkGray
        rgba = [rgb.redComponent, rgb.greenComponent, rgb.blueComponent, rgb.alphaComponent]
    }
}

/// Serialize native segmentation so rapid color changes cannot accumulate
/// simultaneous image-processing jobs. No preview files are written.
actor StudioPhotoPreviewRenderer {
    static let shared = StudioPhotoPreviewRenderer()
    private let context = CIContext(options: [.useSoftwareRenderer: true])

    func render(_ request: StudioPhotoPreviewRequest) throws -> CGImage? {
        try Task.checkCancellation()
        guard !request.photo.isEmpty else { return nil }
        if request.backgroundMode == "Original" {
            guard let source = PhotoPreview.load(path: request.photo) else {
                throw StudioError.message("The outfit photo is unavailable. Select or import it again.")
            }
            return source
        }
        // Match export's Core Image decode before segmentation. An ImageIO
        // thumbnail can change the segmentation at fine clothing/hair edges.
        guard var source = CIImage(contentsOf: URL(fileURLWithPath: request.photo), options: [.applyOrientationProperty: true]),
              !source.extent.isEmpty, !source.extent.isInfinite else {
            throw StudioError.message("The outfit photo is unavailable. Select or import it again.")
        }
        source = source.transformed(by: CGAffineTransform(translationX: -source.extent.minX, y: -source.extent.minY))
        let scale = min(1, 1600 / max(source.extent.width, source.extent.height))
        if scale < 1 { source = source.transformed(by: CGAffineTransform(scaleX: scale, y: scale)) }
        guard request.rgba.count == 4, request.rgba.allSatisfy({ $0.isFinite && (0...1).contains($0) }) else {
            throw StudioError.message("Choose an RGB background color.")
        }
        if request.backgroundMode == "Image" && request.backgroundPath.isEmpty {
            throw StudioError.message("Import a background image to preview it.")
        }
        let color = NSColor(srgbRed: request.rgba[0], green: request.rgba[1], blue: request.rgba[2], alpha: request.rgba[3])
        let result = try PhotoPreparation.compositeImage(input: source,
            background: request.backgroundMode == "Image" ? URL(fileURLWithPath: request.backgroundPath) : nil,
            color: color)
        try Task.checkCancellation()
        guard let image = context.createCGImage(result, from: result.extent, format: .RGBA8,
                                                colorSpace: CGColorSpaceCreateDeviceRGB()) else {
            throw StudioError.message("Unable to preview this background. Choose another image or color.")
        }
        return image
    }
}

struct StudioPhotoPreview: View {
    let request: StudioPhotoPreviewRequest
    @PreviewViewState private var image: CGImage?
    @PreviewViewState private var loading = false
    @PreviewViewState private var error: String?

    var body: some View {
        ZStack {
            if let image {
                Image(image, scale: 1, label: Text(request.backgroundMode == "Original" ? "Original outfit photograph" : "Outfit photograph with selected background"))
                    .resizable().scaledToFit()
            }
            else if !loading && error == nil { ContentUnavailableView("Choose an outfit", systemImage: "person.crop.rectangle") }
            if loading { ProgressView("Updating preview…").padding(12).background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8)) }
            if let error { Text(error).font(.callout).multilineTextAlignment(.center).padding(16).background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8)) }
        }
        .task(id: request) {
            image = nil; error = nil; loading = !request.photo.isEmpty
            do {
                // Coalesce picker drags before starting native segmentation.
                try await Task.sleep(for: .milliseconds(200))
                let rendered = try await StudioPhotoPreviewRenderer.shared.render(request)
                try Task.checkCancellation()
                image = rendered; loading = false
            } catch is CancellationError {
                // A new request or disappearing view owns the next UI state.
            } catch {
                guard !Task.isCancelled else { return }
                self.error = error.localizedDescription; loading = false
            }
        }
    }
}
