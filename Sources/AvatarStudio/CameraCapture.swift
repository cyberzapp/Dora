import SwiftUI
import AVFoundation
import AppKit

protocol CaptureSessionControlling: Sendable {
    var session: AVCaptureSession { get }
    var output: AVCaptureMovieFileOutput { get }
    func start(cameraID: String) async throws
    func record(to url: URL, delegate: CameraCapture)
    func stopRecording()
    func stop()
}

/// Configuration, session start/stop and movie-output mutations are serialized
/// on this private queue. The immutable session is also handed to Apple's
/// preview layer; no UI state is stored or mutated on the capture queue.
private final class CaptureSessionController: CaptureSessionControlling, @unchecked Sendable {
    let session = AVCaptureSession()
    let output = AVCaptureMovieFileOutput()
    private let queue = DispatchQueue(label: "studio.camera")

    func start(cameraID: String) async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            queue.async {
                do {
                    if self.session.isRunning { self.session.stopRunning() }
                    self.session.beginConfiguration()
                    do {
                        defer { self.session.commitConfiguration() }
                        self.session.inputs.forEach { self.session.removeInput($0) }
                        guard let camera = AVCaptureDevice(uniqueID: cameraID), let microphone = AVCaptureDevice.default(for: .audio) else { throw StudioError.message("Connect a camera and microphone.") }
                        for device in [camera, microphone] {
                            let input = try AVCaptureDeviceInput(device: device)
                            guard self.session.canAddInput(input) else { throw StudioError.message("Cannot use the selected capture device") }
                            self.session.addInput(input)
                        }
                        if self.session.canSetSessionPreset(.hd1920x1080) { self.session.sessionPreset = .hd1920x1080 }
                        else if self.session.canSetSessionPreset(.hd1280x720) { self.session.sessionPreset = .hd1280x720 }
                        else { self.session.sessionPreset = .high }
                        if !self.session.outputs.contains(self.output) {
                            guard self.session.canAddOutput(self.output) else { throw StudioError.message("The selected camera cannot record a movie. Choose another camera.") }
                            self.session.addOutput(self.output)
                        }
                    }
                    self.session.startRunning()
                    guard self.session.isRunning else { throw StudioError.message("Camera preview did not start. Check the camera connection and try again.") }
                    continuation.resume()
                } catch { continuation.resume(throwing: error) }
            }
        }
    }
    func record(to url: URL, delegate: CameraCapture) {
        queue.async { self.output.startRecording(to: url, recordingDelegate: delegate) }
    }
    func stopRecording() { queue.async { if self.output.isRecording { self.output.stopRecording() } } }
    func stop() {
        queue.async {
            if self.output.isRecording { self.output.stopRecording() }
            if self.session.isRunning { self.session.stopRunning() }
        }
    }
}

@MainActor final class CameraCapture: NSObject, ObservableObject, AVCaptureFileOutputRecordingDelegate {
    private let controller: any CaptureSessionControlling
    private let requestPermission: @MainActor (AVMediaType) async -> Bool
    var session: AVCaptureSession { controller.session }
    var output: AVCaptureMovieFileOutput { controller.output }
    @Published var devices: [AVCaptureDevice] = []
    @Published var selected = "" { didSet { if selected != oldValue { stop() } } }
    @Published private(set) var ready = false
    @Published private(set) var preparing = false
    @Published private(set) var recording = false
    @Published private(set) var recordingRequested = false
    @Published private(set) var countdown = 0
    @Published private(set) var elapsed = 0
    @Published var error: String?
    private var timer: Timer?
    private var countdownTask: Task<Void, Never>?
    private var operationID = UUID()
    var recordingDirectory = FileManager.default.temporaryDirectory
    var onSaved: ((URL) -> Bool)?

    init(controller: (any CaptureSessionControlling)? = nil,
         requestPermission: @escaping @MainActor (AVMediaType) async -> Bool = { await AVCaptureDevice.requestAccess(for: $0) }) {
        self.controller = controller ?? CaptureSessionController()
        self.requestPermission = requestPermission
        super.init()
    }

    func discover() {
        devices = AVCaptureDevice.DiscoverySession(deviceTypes: [.builtInWideAngleCamera, .external], mediaType: .video, position: .unspecified).devices
        selected = devices.first?.uniqueID ?? ""
    }
    func start() {
        guard !preparing, !recording, !recordingRequested, countdown == 0 else { return }
        guard !selected.isEmpty else { error = "Connect and select a camera first."; return }
        let token = UUID(); operationID = token
        let id = selected
        preparing = true; ready = false; error = nil
        Task { [weak self] in
            guard let self, operationID == token else { return }
            let videoAllowed = await requestPermission(.video)
            guard operationID == token else { return }
            guard videoAllowed else {
                preparing = false; error = "Enable Camera for Avatar Studio in System Settings → Privacy & Security."; return
            }
            let audioAllowed = await requestPermission(.audio)
            guard operationID == token else { return }
            guard audioAllowed else {
                preparing = false; error = "Enable Microphone for Avatar Studio in System Settings → Privacy & Security."; return
            }
            do {
                try await controller.start(cameraID: id)
                guard operationID == token else { return }
                ready = true; preparing = false
            } catch {
                guard operationID == token else { return }
                preparing = false; self.error = error.localizedDescription
            }
        }
    }
    func record() {
        guard ready, !recording, !recordingRequested, countdown == 0 else { return }
        let token = operationID
        countdown = 3
        countdownTask = Task { [weak self] in
            guard let self else { return }
            for value in [3, 2, 1] {
                guard !Task.isCancelled, ready, operationID == token else { return }
                countdown = value
                do { try await Task.sleep(for: .seconds(1)) } catch { return }
            }
            guard !Task.isCancelled, ready, operationID == token else { return }
            do { try FileManager.default.createDirectory(at: recordingDirectory, withIntermediateDirectories: true) }
            catch { countdown = 0; self.error = "Cannot save a camera recording: \(error.localizedDescription)"; return }
            countdown = 0; elapsed = 0; recordingRequested = true
            let url = recordingDirectory.appendingPathComponent("avatar-\(UUID().uuidString).mov")
            controller.record(to: url, delegate: self)
        }
    }
    func stopRecording() { controller.stopRecording() }
    func stop() {
        operationID = UUID(); ready = false; preparing = false; countdown = 0
        countdownTask?.cancel(); countdownTask = nil
        controller.stop()
    }
    nonisolated func fileOutput(_ output: AVCaptureFileOutput, didStartRecordingTo fileURL: URL, from connections: [AVCaptureConnection]) {
        Task { @MainActor [weak self] in
            guard let self else { return }
            guard ready else { controller.stopRecording(); return }
            recordingRequested = false
            recording = true; timer?.invalidate()
            timer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { [weak self] _ in
                Task { @MainActor in
                    guard let self, self.recording else { return }
                    self.elapsed += 1; if self.elapsed >= 30 { self.stopRecording() }
                }
            }
        }
    }
    nonisolated func fileOutput(_ output: AVCaptureFileOutput, didFinishRecordingTo outputFileURL: URL, from connections: [AVCaptureConnection], error: Error?) {
        let message: String?
        if let error, !((error as NSError).userInfo[AVErrorRecordingSuccessfullyFinishedKey] as? Bool ?? false) { message = error.localizedDescription }
        else { message = nil }
        Task { @MainActor [weak self] in
            guard let self else { return }
            recording = false; recordingRequested = false; timer?.invalidate(); timer = nil
            if let message { self.error = "\(message) Recording retained at \(outputFileURL.path)." }
            else if onSaved?(outputFileURL) == true { try? FileManager.default.removeItem(at: outputFileURL) }
            else { self.error = "The recording could not be added to your library. It has been preserved at \(outputFileURL.path). Import that file after resolving the save error." }
        }
    }
}
struct CameraPreview: NSViewRepresentable {
    let session: AVCaptureSession
    func makeNSView(context: Context) -> NSView {
        let view = PreviewView(); view.wantsLayer = true
        let layer = AVCaptureVideoPreviewLayer(session: session); layer.videoGravity = .resizeAspect
        view.layer = layer; return view
    }
    func updateNSView(_ view: NSView, context: Context) {}
    final class PreviewView: NSView { override func layout() { super.layout(); layer?.frame = bounds } }
}
