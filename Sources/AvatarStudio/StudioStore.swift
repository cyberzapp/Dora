import Foundation
import AppKit
import AVFoundation
import UniformTypeIdentifiers

@MainActor final class StudioStore: ObservableObject {
    @Published var library = Library()
    @Published var error: String?
    @Published var busy = false
    @Published var diagnostics = "Run diagnostics to check the local runtime."
    @Published var modelStatus = LocalModelStatus()
    @Published var selectedAvatar = ""
    @Published var selectedOutfit = ""
    @Published var audio = ""
    @Published var motion = ""
    @Published var sourceVideo = ""
    @Published var script = ""
    @Published var projectName = "Untitled video"
    @Published var portrait = false
    @Published var waveform: [Float] = []
    @Published var audioDuration = 0.0
    @Published var consent = false
    @Published var backgroundMode = "Original"
    @Published var backgroundPath = ""
    @Published var backgroundColor = NSColor.darkGray
    @Published var runtime: String { didSet { preferences.set(runtime, forKey: "runtime"); invalidateModelStatus() } }
    @Published var renderEngine: RenderEngine { didSet { preferences.set(renderEngine.rawValue, forKey: "renderEngine"); invalidateModelStatus() } }
    @Published var stabilizeHead: Bool { didSet { preferences.set(stabilizeHead, forKey: "stabilizeHead") } }
    let root: URL
    private let preferences: UserDefaults
    private var db: Database?
    private var process: Process?
    private var renderActivity: NSObjectProtocol?
    private var activeID: String?
    private var activeAttempt: String?
    private var timer: Timer?
    private var shuttingDown = false
    private var lastModelPoll = Date.distantPast
    private var modelStatusTask: Task<Void, Never>?
    private var modelStatusGeneration = 0
    var outfit: Outfit? { library.avatars.first(where: {$0.id == selectedAvatar})?.outfits.first(where: {$0.id == selectedOutfit}) }
    init(storageRoot: URL? = nil, runtimeRoot: String? = nil, preferences: UserDefaults = .standard) {
        self.preferences = preferences
        root = storageRoot ?? FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0].appendingPathComponent("Avatar Studio")
        let resolvedRuntime = runtimeRoot ?? preferences.string(forKey: "runtime") ?? Bundle.main.object(forInfoDictionaryKey: "AvatarRuntimeRoot") as? String ?? FileManager.default.currentDirectoryPath
        runtime = resolvedRuntime
        let installedFlashHead = FileManager.default.fileExists(atPath: resolvedRuntime + "/.local/flashhead-models/download-complete.json") && FileManager.default.fileExists(atPath: resolvedRuntime + "/.local/flashhead-audio/download-complete.json")
        renderEngine = RenderEngine(rawValue: preferences.string(forKey: "renderEngine") ?? "") ?? (installedFlashHead ? .flashHead : .museTalk)
        stabilizeHead = preferences.object(forKey: "stabilizeHead") as? Bool ?? true
        do {
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            for sub in ["originals", "audio", "motions", "jobs"] { try FileManager.default.createDirectory(at: root.appendingPathComponent(sub), withIntermediateDirectories: true) }
            let database = try Database(url: root.appendingPathComponent("studio.sqlite"))
            library = try database.load()
            db = database
            for i in library.jobs.indices where ["Rendering", "Interrupted"].contains(library.jobs[i].status) {
                let job = library.jobs[i]
                let directory = root.appendingPathComponent("jobs/\(job.id)")
                if let request = Self.readJSONObject(directory.appendingPathComponent("request.json")),
                   let attempt = request["attempt"] as? String, UUID(uuidString: attempt) != nil,
                   let cancellation = Self.readJSONObject(directory.appendingPathComponent("cancellation.json")),
                   cancellation["attempt"] as? String == attempt,
                   cancellation["job"] as? String == job.id {
                    library.jobs[i].status = "Cancelled"
                    library.jobs[i].output = ""
                    library.jobs[i].detail = "Recovered your cancellation request"
                    continue
                }
                // A previous attempt's completion can remain on disk until the
                // current worker reports its first status. Never revive it.
                if let request = Self.readJSONObject(directory.appendingPathComponent("request.json")),
                   let attempt = request["attempt"] as? String, UUID(uuidString: attempt) != nil,
                   let value = Self.readJSONObject(directory.appendingPathComponent("status.json")),
                   value["attempt"] as? String == attempt, value["status"] as? String == "Completed",
                   let output = Self.completedOutput(value, job: job, directory: directory) {
                    library.jobs[i].status = "Completed"
                    library.jobs[i].progress = 1
                    library.jobs[i].output = output
                    library.jobs[i].detail = value["detail"] as? String ?? "Recovered completed render"
                }
            }
            for i in library.jobs.indices where ["Rendering", "Queued"].contains(library.jobs[i].status) { library.jobs[i].status = "Interrupted"; library.jobs[i].detail = (library.jobs[i].engine ?? .museTalk).supportsResume ? "Previous session ended. Resume to reuse verified saved progress." : "Previous session ended. Retry to start a new render." }
            selectFirst(); restoreDraft(); save()
        } catch { self.error = error.localizedDescription }
        timer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in Task { @MainActor in self?.poll() } }
    }
    private static func readJSONObject(_ url: URL) -> [String: Any]? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
    }
    private static func completedOutput(_ status: [String: Any], job: RenderJob, directory: URL) -> String? {
        let name: String
        switch job.engine ?? .museTalk {
        case .museTalk: name = "export.mov"
        case .longCat: name = "generated.mov"
        case .flashHead: name = "output.mov"
        case .ditto: name = "ditto.mov"
        }
        let expected = directory.appendingPathComponent(name).standardizedFileURL
        guard let path = status["output"] as? String, path.hasPrefix("/"),
              URL(fileURLWithPath: path).standardizedFileURL.path == expected.path,
              let values = try? expected.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey]),
              values.isRegularFile == true, values.isSymbolicLink == false, (values.fileSize ?? 0) > 0 else { return nil }
        return expected.path
    }
    @discardableResult func save() -> Bool {
        guard let db else { error = "The local library is unavailable. Your existing data has been preserved. Restart after resolving the library error."; return false }
        library.draft = currentDraft()
        do { try db.save(library); return true } catch { self.error = error.localizedDescription; return false }
    }
    private func currentDraft() -> VideoDraft {
        let color = backgroundColor.usingColorSpace(.sRGB) ?? NSColor.darkGray
        return VideoDraft(avatar: selectedAvatar, outfit: selectedOutfit, audio: audio,
                          motion: motion, sourceVideo: sourceVideo, script: script,
                          name: projectName, portrait: portrait, waveform: waveform,
                          audioDuration: audioDuration, backgroundMode: backgroundMode,
                          backgroundPath: backgroundPath,
                          backgroundRGBA: [color.redComponent, color.greenComponent, color.blueComponent, color.alphaComponent],
                          engine: renderEngine)
    }
    private func restoreDraft() {
        guard let draft = library.draft else { return }
        if let avatar = library.avatars.first(where: { $0.id == draft.avatar }) {
            selectedAvatar = avatar.id
            selectedOutfit = avatar.outfits.contains(where: { $0.id == draft.outfit }) ? draft.outfit : (avatar.defaultOutfit ?? avatar.outfits.first?.id ?? "")
        }
        audio = draft.audio; waveform = draft.waveform; audioDuration = draft.audioDuration
        motion = draft.motion; sourceVideo = draft.sourceVideo; script = draft.script
        projectName = draft.name; portrait = draft.portrait
        backgroundMode = draft.backgroundMode; backgroundPath = draft.backgroundPath
        if draft.backgroundRGBA.count == 4, draft.backgroundRGBA.allSatisfy({ $0.isFinite && (0...1).contains($0) }) {
            backgroundColor = NSColor(srgbRed: draft.backgroundRGBA[0], green: draft.backgroundRGBA[1], blue: draft.backgroundRGBA[2], alpha: draft.backgroundRGBA[3])
        }
        renderEngine = draft.engine
        var missing: [String] = []
        if !audio.isEmpty && !FileManager.default.fileExists(atPath: audio) {
            audio = ""; waveform = []; audioDuration = 0; missing.append("voice-over")
        }
        if !motion.isEmpty && !FileManager.default.fileExists(atPath: motion) { motion = ""; missing.append("motion reference") }
        if !sourceVideo.isEmpty && !FileManager.default.fileExists(atPath: sourceVideo) { sourceVideo = ""; missing.append("talking video") }
        if !backgroundPath.isEmpty && !FileManager.default.fileExists(atPath: backgroundPath) {
            backgroundPath = ""; missing.append("background image")
        }
        if !missing.isEmpty { error = "Your draft was restored, but these files are missing: \(missing.joined(separator: ", ")). Import them again to continue." }
    }
    func selectFirst() { if let avatar = library.avatars.first { selectedAvatar = avatar.id; selectedOutfit = avatar.defaultOutfit ?? avatar.outfits.first?.id ?? "" } }
    func pick(_ types: [UTType]) -> URL? {
        let panel = NSOpenPanel(); panel.allowedContentTypes = types; panel.canChooseDirectories = false
        return panel.runModal() == .OK ? panel.url : nil
    }
    func createAvatar(name: String) {
        guard consent, !name.trimmingCharacters(in: .whitespaces).isEmpty else { error = "Enter a name and confirm likeness consent."; return }
        let avatar = Avatar(name: name, consent: true)
        commitAvatarEdit { library.avatars.append(avatar); selectedAvatar = avatar.id; selectedOutfit = "" }
    }
    /// Keep the visible library and its selection consistent with durable storage.
    /// Prepared source files are retained if a save fails; no original is removed.
    private func commitAvatarEdit(_ edit: () -> Void) {
        let previous = library, previousAvatar = selectedAvatar, previousOutfit = selectedOutfit
        edit()
        if !save() { library = previous; selectedAvatar = previousAvatar; selectedOutfit = previousOutfit }
    }
    private func commitPreparedOutfit(_ prepared: Outfit, avatarID: String, replacing oldID: String?) throws {
        guard let index = library.avatars.firstIndex(where: { $0.id == avatarID }), library.avatars[index].consent else {
            throw StudioError.message("The avatar is no longer available. The prepared source files are retained locally.")
        }
        var result = prepared
        if let oldID {
            guard let previous = library.avatars[index].outfits.first(where: { $0.id == oldID }) else {
                throw StudioError.message("The outfit being replaced is no longer available. Import the photograph as a new outfit.")
            }
            result.name = previous.name
        }
        commitAvatarEdit {
            if let oldID, let outfitIndex = library.avatars[index].outfits.firstIndex(where: { $0.id == oldID }) {
                library.avatars[index].outfits[outfitIndex] = result
            } else { library.avatars[index].outfits.append(result) }
            if library.avatars[index].defaultOutfit == nil || (oldID != nil && library.avatars[index].defaultOutfit == oldID) {
                library.avatars[index].defaultOutfit = result.id
            }
            library.avatars[index].updated = Date()
            if selectedAvatar == avatarID { selectedOutfit = result.id }
        }
    }
    func importPhoto(replace: Bool = false) {
        guard !busy else { return }
        guard library.avatars.first(where: { $0.id == selectedAvatar })?.consent == true else {
            error = "Create and select a consented avatar first."; return
        }
        guard let source = pick([.jpeg, .png, .heic]) else { return }
        importPhoto(from: source, replace: replace)
    }
    func importPhoto(from source: URL, replace: Bool = false) {
        guard !busy else { return }
        guard let index = library.avatars.firstIndex(where: {$0.id == selectedAvatar}), library.avatars[index].consent else { error = "Create and select a consented avatar first."; return }
        guard !replace || library.avatars[index].outfits.contains(where: { $0.id == selectedOutfit }) else {
            error = "Select an outfit before replacing its photo."; return
        }
        let avatarID = selectedAvatar, oldID = selectedOutfit, directory = root.appendingPathComponent("originals")
        busy = true; error = nil
        Task {
            do {
                let result = try await Task.detached { try PhotoPreparation.prepare(source: source, directory: directory) }.value
                try commitPreparedOutfit(result, avatarID: avatarID, replacing: replace ? oldID : nil)
            } catch { self.error = error.localizedDescription }
            busy = false
        }
    }
    func renameOutfit(_ name: String) {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, let i = library.avatars.firstIndex(where: {$0.id == selectedAvatar}), let j = library.avatars[i].outfits.firstIndex(where: {$0.id == selectedOutfit}) else { return }
        commitAvatarEdit { library.avatars[i].outfits[j].name = trimmed; library.avatars[i].updated = Date() }
    }
    func importVideoFrame(source: URL, seconds: Double) {
        guard !busy, let avatar = library.avatars.first(where: {$0.id == selectedAvatar}), avatar.consent else {
            error = "Select an avatar with likeness consent before importing a video frame."; return
        }
        let avatarID = avatar.id, directory = root.appendingPathComponent("originals")
        busy = true; error = nil
        Task {
            do {
                let outfit = try await Task.detached { try await VideoFramePreparation.prepare(source: source, seconds: seconds, directory: directory) }.value
                try commitPreparedOutfit(outfit, avatarID: avatarID, replacing: nil)
            } catch { self.error = error.localizedDescription }
            busy = false
        }
    }
    func defaultOutfit() {
        guard let i = library.avatars.firstIndex(where: {$0.id == selectedAvatar}),
              library.avatars[i].outfits.contains(where: { $0.id == selectedOutfit }) else { return }
        commitAvatarEdit { library.avatars[i].defaultOutfit = selectedOutfit; library.avatars[i].updated = Date() }
    }
    func deleteOutfit() {
        guard let i = library.avatars.firstIndex(where: {$0.id == selectedAvatar}),
              library.avatars[i].outfits.contains(where: {$0.id == selectedOutfit}) else { return }
        commitAvatarEdit {
            library.avatars[i].outfits.removeAll(where: {$0.id == selectedOutfit})
            if library.avatars[i].defaultOutfit == selectedOutfit { library.avatars[i].defaultOutfit = library.avatars[i].outfits.first?.id }
            selectedOutfit = library.avatars[i].defaultOutfit ?? ""
            library.avatars[i].updated = Date()
        }
    }
    func importAudio() {
        guard !busy, let url = pick([.audio]) else { return }
        importAudio(from: url)
    }
    func importAudio(from source: URL) {
        guard !busy else { return }
        busy = true; error = nil
        let directory = root.appendingPathComponent("audio")
        Task {
            do {
                let result = try await Task.detached { try AudioPreparation.prepare(source: source, directory: directory) }.value
                audio = result.path; waveform = result.waveform; audioDuration = result.duration
                save()
            } catch { self.error = error.localizedDescription }
            busy = false
        }
    }
    func importMotion() { if let url = pick([.movie]) { storeMotion(url) } }
    @discardableResult func storeMotion(_ source: URL) -> Bool {
        do { let destination = root.appendingPathComponent("motions/\(UUID().uuidString).mov"); try FileManager.default.copyItem(at: source, to: destination); library.motions.append(destination.path); motion = destination.path; return save() } catch { self.error = error.localizedDescription; return false }
    }
    static let testClipSeconds: Double = 5
    func enqueue(testClip: Bool = false) {
        guard !busy else { return }
        guard db != nil else { error = "Resolve the local library error before creating a render."; return }
        guard library.avatars.first(where: {$0.id == selectedAvatar})?.consent == true else { error = "Select an avatar with likeness consent."; return }
        guard let outfit, !audio.isEmpty else { error = "Choose an outfit photo and import a voice-over."; return }
        guard !testClip || renderEngine == .flashHead else { error = "Short test clips are available with FlashHead."; return }
        if renderEngine == .ditto {
            guard audioDuration > 0, audioDuration <= 15 else { error = "Choose a voice-over up to 15 seconds for Ditto evaluation."; return }
            guard motion.isEmpty, sourceVideo.isEmpty else { error = "Clear the reference video to use Ditto with the outfit photo."; return }
        } else if renderEngine.usesAutomaticMotion {
            guard audioDuration >= (renderEngine == .flashHead ? 0.5 : 1), audioDuration <= 300 else { error = "Choose a voice-over up to five minutes long, with at least one second of speech."; return }
            guard motion.isEmpty, sourceVideo.isEmpty else { error = "This engine uses the outfit photo and automatic motion. Clear the reference-video choices first."; return }
            let markers = renderEngine == .flashHead ? ["flashhead-models", "flashhead-audio"] : ["longcat-models/q4"]
            guard markers.allSatisfy({FileManager.default.fileExists(atPath: runtime + "/.local/\($0)/download-complete.json")}) else { error = "The selected engine's model setup is incomplete. Check Settings for progress."; return }
        }
        busy = true; error = nil
        let bgMode = backgroundMode, bgPath = backgroundPath, color = backgroundColor
        let capturedAudio = audio, capturedMotion = motion, capturedScript = script, capturedName = projectName, capturedPortrait = portrait, capturedVideo = sourceVideo
        let capturedEngine = renderEngine
        let capturedStabilization = stabilizeHead
        var capturedFaceBounds: [Double]?
        if capturedEngine.usesAutomaticMotion && capturedEngine != .ditto {
            let metadata = URL(fileURLWithPath: outfit.prepared).deletingLastPathComponent().appendingPathComponent("\(outfit.id)-landmarks.json")
            if let data = try? Data(contentsOf: metadata), let value = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                capturedFaceBounds = value["face"] as? [Double]
            }
            guard capturedFaceBounds?.count == 4 else { error = "Reimport this outfit photo to prepare source-body preservation."; busy = false; return }
        }
        let jobID = UUID().uuidString
        let directory = root.appendingPathComponent("jobs/\(jobID)")
        Task {
            do {
                try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
                var photo = outfit.prepared
                if capturedEngine == .flashHead || capturedEngine == .ditto {
                    let target = directory.appendingPathComponent("source-full-resolution.png")
                    try await Task.detached { try PhotoPreparation.renderSource(source: URL(fileURLWithPath: outfit.original), output: target) }.value
                    photo = target.path
                }
                if bgMode != "Original" && capturedVideo.isEmpty {
                    if bgMode == "Image" && bgPath.isEmpty { throw StudioError.message("Import a background image first.") }
                    let target = directory.appendingPathComponent("composite.png")
                    let sourcePhoto = photo
                    try await Task.detached { try PhotoPreparation.composite(photo: URL(fileURLWithPath: sourcePhoto), background: bgMode == "Image" ? URL(fileURLWithPath: bgPath) : nil, color: color, output: target) }.value
                    photo = target.path
                }
                var jobAudio = capturedAudio
                if testClip {
                    let target = directory.appendingPathComponent("test-clip.wav")
                    let seconds = Self.testClipSeconds
                    try await Task.detached {
                        try AudioPreparation.writePreview(source: URL(fileURLWithPath: capturedAudio), output: target, maximumSeconds: seconds)
                    }.value
                    jobAudio = target.path
                }
                library.jobs.append(RenderJob(id: jobID, name: testClip ? "\(capturedName) • Test clip" : capturedName, photo: photo, audio: jobAudio, motion: capturedMotion, script: capturedScript, sourceVideo: capturedVideo.isEmpty ? nil : capturedVideo, portrait: capturedPortrait, engine: capturedEngine, faceBounds: capturedFaceBounds, stabilizeHead: capturedEngine == .flashHead ? capturedStabilization : nil))
                if save() { startNext() } else {
                    library.jobs.removeAll(where: {$0.id == jobID})
                    try? FileManager.default.removeItem(at: directory)
                }
            } catch {
                try? FileManager.default.removeItem(at: directory)
                self.error = error.localizedDescription
            }
            busy = false
        }
    }
    func review(_ job: RenderJob, decision: RenderReviewDecision?) {
        guard let i = library.jobs.firstIndex(where: {$0.id == job.id}), library.jobs[i].status == "Completed" else { return }
        let previous = library.jobs[i].review
        library.jobs[i].review = decision.map { RenderReview(decision: $0) }
        if !save() { library.jobs[i].review = previous }
    }
    func retry(_ job: RenderJob, resume: Bool = true) {
        guard let index = library.jobs.firstIndex(where: { $0.id == job.id }) else { return }
        let current = library.jobs[index]
        guard ["Failed", "Interrupted", "Cancelled", "Completed"].contains(current.status) else { return }
        if resume, (current.engine ?? .museTalk).supportsResume, ["Failed", "Interrupted", "Cancelled"].contains(current.status) {
            guard activeID != job.id else { error = "Wait for cancellation to finish before resuming this render."; return }
            library.jobs[index].status = "Queued"; library.jobs[index].detail = "Resuming verified saved progress"
            library.jobs[index].output = ""
            guard save() else { library.jobs[index] = current; return }
            // The durable queue request supersedes an older cancellation even
            // while another job is running. A failed removal leaves the retry
            // available, but must not start an ambiguously cancelled attempt.
            let cancellation = root.appendingPathComponent("jobs/\(job.id)/cancellation.json")
            do {
                if FileManager.default.fileExists(atPath: cancellation.path) { try FileManager.default.removeItem(at: cancellation) }
            } catch {
                library.jobs[index] = current; save()
                self.error = "Could not clear the previous cancellation. Check access to the render folder before retrying."
                return
            }
            startNext(); return
        }
        var copy = current; copy.id = UUID().uuidString; copy.status = "Queued"; copy.progress = 0; copy.output = ""; copy.created = Date(); copy.review = nil
        library.jobs.append(copy)
        guard save() else { library.jobs.removeAll(where: { $0.id == copy.id }); return }
        startNext()
    }
    func cancel(_ job: RenderJob) {
        guard let i = library.jobs.firstIndex(where: {$0.id == job.id}) else { return }
        guard ["Queued", "Rendering"].contains(library.jobs[i].status) else { return }
        var journalError: Error?
        if job.id == activeID {
            let directory = root.appendingPathComponent("jobs/\(job.id)")
            if let attempt = activeAttempt {
                do {
                    try JSONSerialization.data(withJSONObject: ["job":job.id,"attempt":attempt]).write(
                        to: directory.appendingPathComponent("cancellation.json"), options: .atomic)
                } catch { journalError = error }
            }
            FileManager.default.createFile(atPath: directory.appendingPathComponent("cancel").path, contents: Data())
            process?.terminate()
        }
        library.jobs[i].status = "Cancelled"; library.jobs[i].detail = "Cancelled by you"
        if !save(), journalError != nil {
            error = "Cancellation was requested, but neither its recovery record nor the library could be saved. Check free disk space and access to the workspace."
        }
    }
    private func startNext() {
        guard !shuttingDown, process == nil, let job = library.jobs.first(where: {$0.status == "Queued"}), let i = library.jobs.firstIndex(where: {$0.id == job.id}) else { return }
        do {
            let directory = root.appendingPathComponent("jobs/\(job.id)")
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            try? FileManager.default.removeItem(at: directory.appendingPathComponent("cancel"))
            let cancellation = directory.appendingPathComponent("cancellation.json")
            if FileManager.default.fileExists(atPath: cancellation.path) { try FileManager.default.removeItem(at: cancellation) }
            let longCat = job.engine == .longCat
            let engine = job.engine ?? .museTalk
            let size = engine.outputSize(portrait: job.portrait)
            let attempt = UUID().uuidString
            let requestURL = directory.appendingPathComponent("request.json")
            if let previous = try? Data(contentsOf: requestURL) {
                let snapshot = try? JSONSerialization.jsonObject(with: previous) as? [String: Any]
                let oldAttempt = (snapshot?["attempt"] as? String).flatMap(UUID.init(uuidString:))?.uuidString ?? UUID().uuidString
                let archive = directory.appendingPathComponent("attempts/\(oldAttempt)")
                try FileManager.default.createDirectory(at: archive, withIntermediateDirectories: true)
                // Previous inputs and errors remain reviewable after a retry.
                try previous.write(to: archive.appendingPathComponent("request.json"), options: .atomic)
                for name in ["worker.log", "status.json", "launch.json"] {
                    let source = directory.appendingPathComponent(name)
                    if FileManager.default.fileExists(atPath: source.path) {
                        let target = archive.appendingPathComponent(name)
                        if !FileManager.default.fileExists(atPath: target.path) { try FileManager.default.copyItem(at: source, to: target) }
                    }
                }
            }
            let request: [String: Any] = ["attempt":attempt, "photo":job.photo, "audio":job.audio, "motion":job.motion, "source_video":job.sourceVideo ?? "", "portrait":job.portrait, "consent":true, "directory":directory.path, "engine":engine.rawValue, "width":size.width, "height":size.height, "face_bounds":job.faceBounds ?? [], "preserve_source_body":true, "stabilize_head":job.stabilizeHead ?? false, "network_policy":"os-denied"]
            try JSONSerialization.data(withJSONObject: request).write(to: requestURL, options: .atomic)
            let worker = Process(); worker.executableURL = URL(fileURLWithPath: runtime).appendingPathComponent(longCat ? ".venv-longcat/bin/python" : ".venv/bin/python")
            let requestPath = directory.appendingPathComponent("request.json").path
            switch engine {
            case .longCat: worker.arguments = [runtime + "/backend/longcat_render.py", requestPath]
            case .flashHead: worker.arguments = [runtime + "/backend/flashhead_render.py", requestPath]
            case .museTalk, .ditto: worker.arguments = [runtime + "/backend/worker.py", "render", requestPath]
            }
            let isolation = URL(fileURLWithPath: "/usr/bin/sandbox-exec")
            guard FileManager.default.isExecutableFile(atPath: isolation.path), let python = worker.executableURL else {
                throw StudioError.message("This macOS installation cannot start a network-isolated local worker. Run local setup diagnostics before rendering.")
            }
            let networkProfile = "(version 1) (allow default) (deny network*)"
            worker.arguments = ["-p", networkProfile, python.path] + (worker.arguments ?? [])
            worker.executableURL = isolation
            try JSONSerialization.data(withJSONObject: ["attempt":attempt, "worker_network_access":"denied", "launcher":isolation.path, "profile":networkProfile], options: .prettyPrinted).write(to: directory.appendingPathComponent("launch.json"), options: .atomic)
            worker.environment = ProcessInfo.processInfo.environment.merging(["PYTORCH_ENABLE_MPS_FALLBACK":"1", "HF_HUB_OFFLINE":"1", "TRANSFORMERS_OFFLINE":"1", "PATH":"/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"]) { _, new in new }
            let log = directory.appendingPathComponent("worker.log"); FileManager.default.createFile(atPath: log.path, contents: Data())
            let handle = try FileHandle(forWritingTo: log); worker.standardOutput = handle; worker.standardError = handle
            worker.terminationHandler = { [weak self] p in
                try? handle.close()
                Task { @MainActor in
                    guard let self else { return }; self.poll()
                    if let index = self.library.jobs.firstIndex(where: {$0.id == job.id}), self.library.jobs[index].status == "Rendering" { self.library.jobs[index].status = "Failed"; self.library.jobs[index].detail = "Worker exited (\(p.terminationStatus)). Open the log for details." }
                    self.endRenderActivity()
                    self.process = nil; self.activeID = nil; self.activeAttempt = nil
                    if self.save() { self.startNext() }
                }
            }
            library.jobs[i].status = "Rendering"; library.jobs[i].output = ""
            guard save() else { throw StudioError.message("Could not save this render before starting its worker.") }
            try worker.run(); process = worker; activeID = job.id; activeAttempt = attempt
            // User-initiated work prevents idle system sleep, while allowing
            // the display to sleep. End it when this actual worker exits.
            renderActivity = ProcessInfo.processInfo.beginActivity(options: .userInitiated, reason: "Avatar Studio rendering")
        } catch {
            library.jobs[i].status = "Failed"; library.jobs[i].detail = error.localizedDescription
            if save() { startNext() }
        }
    }
    private func endRenderActivity() {
        guard let activity = renderActivity else { return }
        renderActivity = nil
        ProcessInfo.processInfo.endActivity(activity)
    }
    private func poll() {
        if db != nil, library.draft != currentDraft() { save() }
        if Date().timeIntervalSince(lastModelPoll) >= 2 {
            lastModelPoll = Date()
            refreshModelStatus()
        }
        guard let id = activeID, let i = library.jobs.firstIndex(where: {$0.id == id}),
              library.jobs[i].status == "Rendering", let attempt = activeAttempt else { return }
        let directory = root.appendingPathComponent("jobs/\(id)")
        guard let value = Self.readJSONObject(directory.appendingPathComponent("status.json")),
              value["attempt"] as? String == attempt, let status = value["status"] as? String,
              ["Rendering", "Completed", "Failed", "Cancelled"].contains(status) else { return }
        let progress = value["progress"] as? Double ?? 0
        if status == "Completed" {
            guard let output = Self.completedOutput(value, job: library.jobs[i], directory: directory) else { return }
            library.jobs[i].output = output
        }
        library.jobs[i].progress = status == "Completed" ? 1 : (progress.isFinite ? min(1, max(0, progress)) : 0)
        library.jobs[i].detail = value["detail"] as? String ?? ""
        library.jobs[i].status = status
        save()
    }
    private func invalidateModelStatus() {
        modelStatusTask?.cancel(); modelStatusTask = nil
        modelStatusGeneration += 1; lastModelPoll = .distantPast
    }
    func refreshModelStatus() {
        guard !shuttingDown, modelStatusTask == nil else { return }
        let base = runtime, engine = renderEngine, generation = modelStatusGeneration
        modelStatusTask = Task { [weak self] in
            let status = await Task.detached(priority: .utility) {
                LocalModelStatus.read(runtime: base, engine: engine)
            }.value
            guard let self, !Task.isCancelled, !self.shuttingDown, self.modelStatusGeneration == generation else { return }
            self.modelStatus = status; self.modelStatusTask = nil
        }
    }
    func diagnose() {
        let base = runtime, engine = renderEngine; diagnostics = "Checking local runtime…"
        Task {
            diagnostics = await Task.detached {
                let p = Process(); p.executableURL = URL(fileURLWithPath: base).appendingPathComponent(engine == .longCat ? ".venv-longcat/bin/python" : ".venv/bin/python")
                switch engine {
                case .longCat: p.arguments = [base + "/backend/longcat_doctor.py"]
                case .flashHead: p.arguments = [base + "/backend/flashhead_doctor.py"]
                case .ditto: p.arguments = [base + "/backend/ditto_render.py"]
                case .museTalk: p.arguments = [base + "/backend/worker.py", "doctor"]
                }
                let pipe = Pipe(); p.standardOutput = pipe; p.standardError = pipe
                do { try p.run(); let data = pipe.fileHandleForReading.readDataToEndOfFile(); p.waitUntilExit(); return String(decoding: data, as: UTF8.self) } catch { return error.localizedDescription }
            }.value
        }
    }
    func shutdown() {
        shuttingDown = true
        if db != nil { save() }
        timer?.invalidate(); timer = nil
        modelStatusTask?.cancel(); modelStatusTask = nil
        if let id = activeID, let job = library.jobs.first(where: {$0.id == id}) { cancel(job) }
        endRenderActivity()
    }
    func revealLog(_ job: RenderJob) { NSWorkspace.shared.open(root.appendingPathComponent("jobs/\(job.id)/worker.log")) }
    func export(_ job: RenderJob) {
        let panel = NSSavePanel(); panel.allowedContentTypes = [.quickTimeMovie]; panel.nameFieldStringValue = job.name + ".mov"
        if panel.runModal() == .OK, let url = panel.url {
            let temporary = url.deletingLastPathComponent().appendingPathComponent(".avatar-export-\(UUID().uuidString).mov")
            do {
                try FileManager.default.copyItem(atPath: job.output, toPath: temporary.path)
                if FileManager.default.fileExists(atPath: url.path) { _ = try FileManager.default.replaceItemAt(url, withItemAt: temporary) }
                else { try FileManager.default.moveItem(at: temporary, to: url) }
            } catch { try? FileManager.default.removeItem(at: temporary); self.error = error.localizedDescription }
        }
    }
}
