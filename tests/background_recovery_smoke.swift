import Foundation
import AppKit

@main struct BackgroundRecoverySmoke {
    static func main() throws {
        let root = URL(fileURLWithPath: CommandLine.arguments[1]).appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let source = URL(fileURLWithPath: CommandLine.arguments[2])
        let original = try Data(contentsOf: source)
        let bad = root.appendingPathComponent("damaged-background.jpg")
        try Data("Not a valid image".utf8).write(to: bad)
        let output = root.appendingPathComponent("existing-output.png")
        let previous = Data("Existing output fixture".utf8)
        try previous.write(to: output)
        for background in [bad, root.appendingPathComponent("missing.png")] {
            do {
                try PhotoPreparation.composite(photo: source, background: background, color: .darkGray, output: output)
                fatalError("Invalid custom background was silently replaced")
            } catch {
                precondition(error.localizedDescription.contains("background image"))
            }
            let retained = try Data(contentsOf: output)
            precondition(retained == previous, "Failed composition overwrote an existing output")
        }
        let untouched = try Data(contentsOf: source)
        precondition(untouched == original)
        print("Invalid custom backgrounds fail explicitly and preserve prior output/source: passed")
    }
}
