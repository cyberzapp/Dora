import Foundation
import AVFoundation

@main struct CameraSaveRecoverySmoke {
    @MainActor static func main() async throws {
        let directory = URL(fileURLWithPath: CommandLine.arguments[1]).appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appendingPathComponent("capture-fixture.mov")
        let payload = Data("Save recovery fixture; not generated media".utf8)
        try payload.write(to: file)
        let camera = CameraCapture()
        camera.onSaved = { _ in false }
        camera.fileOutput(camera.output, didFinishRecordingTo: file, from: [], error: nil)
        try await Task.sleep(for: .milliseconds(100))
        let retained = try Data(contentsOf: file)
        precondition(retained == payload, "Failed library save discarded the recording")
        precondition(camera.error?.contains(file.path) == true, "Recovery location was hidden")
        var delivered: URL?
        camera.onSaved = { delivered = $0; return true }
        camera.fileOutput(camera.output, didFinishRecordingTo: file, from: [], error: nil)
        try await Task.sleep(for: .milliseconds(100))
        precondition(delivered == file && !FileManager.default.fileExists(atPath: file.path))
        print("Failed capture save retains recording; successful save cleans temporary file: passed")
    }
}
