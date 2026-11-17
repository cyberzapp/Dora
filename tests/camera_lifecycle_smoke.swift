import Foundation
import AVFoundation

/// No camera, microphone, recording or generated media is used by this fixture.
private final class CaptureFixture: CaptureSessionControlling, @unchecked Sendable {
    let session = AVCaptureSession()
    let output = AVCaptureMovieFileOutput()
    private let lock = NSLock()
    private var starts = 0, recordings = 0
    var startCount: Int { lock.withLock { starts } }
    var recordCount: Int { lock.withLock { recordings } }
    func start(cameraID: String) async throws { lock.withLock { starts += 1 } }
    func record(to url: URL, delegate: CameraCapture) { lock.withLock { recordings += 1 } }
    func stopRecording() {}
    func stop() {}
}

@main struct CameraLifecycleSmoke {
    @MainActor static func waitReady(_ camera: CameraCapture) async throws {
        let deadline = Date().addingTimeInterval(2)
        while !camera.ready && Date() < deadline { try await Task.sleep(for: .milliseconds(10)) }
        precondition(camera.ready)
    }
    @MainActor static func main() async throws {
        let deniedController = CaptureFixture()
        let denied = CameraCapture(controller: deniedController, requestPermission: { _ in false })
        denied.selected = "fixture"
        denied.start()
        try await Task.sleep(for: .milliseconds(50))
        precondition(!denied.ready && !denied.preparing && denied.error?.contains("Camera") == true)
        precondition(deniedController.startCount == 0)

        let delayedController = CaptureFixture()
        let delayed = CameraCapture(controller: delayedController, requestPermission: { _ in
            try? await Task.sleep(for: .milliseconds(100)); return true
        })
        delayed.selected = "fixture"; delayed.start(); delayed.stop()
        try await Task.sleep(for: .milliseconds(250))
        precondition(!delayed.ready && delayedController.startCount == 0, "Permission completion revived a stopped preview")

        let controller = CaptureFixture()
        let camera = CameraCapture(controller: controller, requestPermission: { _ in true })
        camera.selected = "fixture"; camera.start(); try await waitReady(camera)
        camera.record(); precondition(camera.countdown == 3)
        try await Task.sleep(for: .milliseconds(200))
        camera.stop()
        camera.start(); try await waitReady(camera)
        // An earlier countdown must not record after a rapid preview restart.
        try await Task.sleep(for: .milliseconds(3100))
        precondition(controller.recordCount == 0 && camera.countdown == 0)
        camera.record(); camera.record()
        try await Task.sleep(for: .milliseconds(3200))
        precondition(controller.recordCount == 1 && camera.recordingRequested, "Duplicate recording request")
        camera.stop()
        print("Camera denied/stale permissions, countdown cancellation and duplicate request guards: passed")
    }
}
