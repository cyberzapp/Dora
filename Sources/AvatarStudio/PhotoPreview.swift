import Foundation
import ImageIO

enum PhotoPreview {
    /// Bound only the UI cache. Render preparation retains full source pixels.
    static func load(path: String) -> CGImage? {
        guard !path.isEmpty, let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL,
            [kCGImageSourceShouldCache: false] as CFDictionary) else { return nil }
        return CGImageSourceCreateThumbnailAtIndex(source, 0, [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: 1600,
            kCGImageSourceShouldCacheImmediately: true
        ] as CFDictionary)
    }
}
