import Foundation
import CryptoKit

@main struct PhotoPreviewSmoke {
    static func main() throws {
        let source = URL(fileURLWithPath: CommandLine.arguments[1])
        let original = SHA256.hash(data: try Data(contentsOf: source))
        guard let preview = PhotoPreview.load(path: source.path) else { fatalError("Preview could not decode real source") }
        precondition(max(preview.width, preview.height) <= 1600)
        precondition(preview.width > 0 && preview.height > 0)
        precondition(PhotoPreview.load(path: "/nonexistent/outfit.png") == nil)
        let final = SHA256.hash(data: try Data(contentsOf: source))
        precondition(original == final)
        print("Bounded real-photo preview and unchanged source: passed")
    }
}
