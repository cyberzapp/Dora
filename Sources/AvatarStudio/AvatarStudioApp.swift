import SwiftUI
import AVKit

// Disambiguate the property wrapper from the SDK 27 State macro.
typealias ViewState<Value> = SwiftUI.State<Value>

@main struct AvatarStudioApp: App {
    @StateObject private var store = StudioStore()
    @AppStorage("appearance") private var appearance = "Dark"
    var body: some Scene {
        WindowGroup {
            ContentView().environmentObject(store).frame(minWidth: 1050, minHeight: 700)
                .preferredColorScheme(appearance == "System" ? nil : appearance == "Dark" ? .dark : .light)
                .onReceive(NotificationCenter.default.publisher(for: NSApplication.willTerminateNotification)) { _ in store.shutdown() }
                .onAppear { NSApplication.shared.setActivationPolicy(.regular); NSApplication.shared.activate(ignoringOtherApps: true) }
        }.defaultSize(width: 1380, height: 880)
        .commands {
            CommandGroup(after: .newItem) { Button("Import outfit photo…") { store.importPhoto() }.keyboardShortcut("i"); Button("Import voice-over…") { store.importAudio() }.keyboardShortcut("i", modifiers: [.command, .shift]) }
        }
        Settings { SettingsView().environmentObject(store).padding(24).frame(width: 650, height: 480) }
    }
}
enum Page: String, CaseIterable, Identifiable {
    case dashboard = "Dashboard", avatars = "Avatar Library", motion = "Motion Studio", create = "Create Video", projects = "Projects", queue = "Render Queue", settings = "Settings"
    var id: String { rawValue }
    var icon: String { switch self { case .dashboard: "square.grid.2x2"; case .avatars: "person.crop.rectangle.stack"; case .motion: "video"; case .create: "play.rectangle"; case .projects: "folder"; case .queue: "list.bullet.rectangle"; case .settings: "gearshape" } }
}
struct ContentView: View {
    @EnvironmentObject var store: StudioStore
    @ViewState<Page?> private var page: Page? = .dashboard
    var body: some View {
        NavigationSplitView {
            VStack(alignment: .leading, spacing: 24) {
                Label("Avatar Studio", systemImage: "person.crop.square.filled.and.at.rectangle").font(.title3.bold()).padding(.horizontal, 12).padding(.top, 20)
                List(Page.allCases, selection: $page) { item in Label(item.rawValue, systemImage: item.icon).padding(.vertical, 6).tag(item) }.listStyle(.sidebar)
                VStack(alignment: .leading, spacing: 8) { Label("Local workspace", systemImage: "internaldrive").font(.callout.bold()); Text("Your likeness. Your Mac.").font(.caption).foregroundStyle(.secondary) }.padding(18)
            }.navigationSplitViewColumnWidth(min: 190, ideal: 220)
        } detail: {
            Group {
                switch page ?? .dashboard {
                case .dashboard: DashboardView(page: $page)
                case .avatars: AvatarLibraryView()
                case .motion: MotionView()
                case .create: VideoStudioView()
                case .projects: JobsView(projects: true)
                case .queue: JobsView(projects: false)
                case .settings: SettingsView().padding(28)
                }
            }.navigationTitle((page ?? .dashboard).rawValue)
        }
        .tint(.indigo)
        .alert("Avatar Studio", isPresented: Binding(get: {store.error != nil}, set: { if !$0 {store.error = nil} })) { Button("OK") { store.error = nil } } message: { Text(store.error ?? "") }
    }
}
struct StudioCard<Content: View>: View {
    let title: String
    @ViewBuilder var content: Content
    var body: some View { VStack(alignment: .leading, spacing: 16) { Text(title).font(.headline); content }.padding(20).frame(maxWidth: .infinity, alignment: .leading).background(.quaternary.opacity(0.35), in: RoundedRectangle(cornerRadius: 14)) }
}
struct DashboardView: View {
    @EnvironmentObject var store: StudioStore
    @Binding var page: Page?
    var body: some View {
        ScrollView { VStack(alignment: .leading, spacing: 26) {
            Text("Your next lesson starts here.").font(.largeTitle.bold())
            Text("Create a presenter video from your real photograph and recorded voice.").font(.title3).foregroundStyle(.secondary)
            HStack(spacing: 18) {
                StudioCard(title: "Avatar library") { Text("\(store.library.avatars.count)").font(.system(size: 38, weight: .semibold)); Button("Manage avatars", systemImage: "person.crop.rectangle") { page = .avatars } }
                StudioCard(title: "Completed videos") { Text("\(store.library.jobs.filter {$0.status == "Completed"}.count)").font(.system(size: 38, weight: .semibold)); Button("Open projects", systemImage: "folder") { page = .projects } }
                StudioCard(title: "Rendering locally") { Text("\(store.library.jobs.filter {["Queued", "Rendering"].contains($0.status)}.count)").font(.system(size: 38, weight: .semibold)); Button("View queue", systemImage: "list.bullet") { page = .queue } }
            }
            StudioCard(title: "Start with your own likeness") {
                Label("1. Import a seated, upper-body photograph", systemImage: "person.crop.rectangle")
                Label("2. Add your recorded voice-over", systemImage: "waveform")
                Label("3. Render and review the actual local result", systemImage: "film")
                Button("Create a video", systemImage: "plus") { page = .create }.buttonStyle(.borderedProminent).controlSize(.large)
            }
            StudioCard(title: "Runtime readiness") { Text("Model installation and a successful hardware test are required. A photo preview is never a generated result.").foregroundStyle(.secondary); Button("Check local setup") { page = .settings; store.diagnose() } }
        }.padding(32) }
    }
}
struct AvatarLibraryView: View {
    @EnvironmentObject var store: StudioStore
    @ViewState<String> private var name = "My avatar"
    @ViewState<String> private var outfitName = ""
    @ViewState<Bool> private var delete = false
    @ViewState<URL?> private var frameVideo = nil
    @ViewState<Bool> private var chooseFrame = false
    var body: some View {
        ScrollView { VStack(alignment: .leading, spacing: 22) {
            HStack { Text("Real photos. Consistent identity.").font(.title.bold()); Spacer(); if store.busy { ProgressView().controlSize(.small) } }
            StudioCard(title: "Create an avatar") {
                HStack { TextField("Display name", text: $name); Button("Create profile") { store.createAvatar(name: name) }.disabled(!store.consent) }
                Toggle("This is my likeness, or I have explicit permission from this person.", isOn: $store.consent)
            }
            if !store.library.avatars.isEmpty {
                Picker("Avatar", selection: $store.selectedAvatar) { ForEach(store.library.avatars) { Text($0.name).tag($0.id) } }.onChange(of: store.selectedAvatar) { _, id in store.selectedOutfit = store.library.avatars.first(where: {$0.id == id})?.defaultOutfit ?? "" }
                HStack { Button("Import outfit photo…", systemImage: "plus") { store.importPhoto() }.buttonStyle(.borderedProminent); Text("JPEG, PNG, HEIC • originals are kept unchanged").foregroundStyle(.secondary) }.disabled(store.busy)
                Button("Choose a frame from video…", systemImage: "film") {
                    if let source = store.pick([.movie]) { frameVideo = source; chooseFrame = true }
                }.disabled(store.busy)
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 185, maximum: 260))], spacing: 18) {
                    ForEach(store.library.avatars.first(where: {$0.id == store.selectedAvatar})?.outfits ?? []) { outfit in
                        Button { store.selectedOutfit = outfit.id; outfitName = outfit.name } label: {
                            VStack(alignment: .leading) { PhotoView(path: outfit.prepared).frame(height: 210).clipped(); Text(outfit.name).font(.headline); Label("Prepared", systemImage: "checkmark.circle").font(.caption).foregroundStyle(.secondary) }.padding(10).background(store.selectedOutfit == outfit.id ? Color.indigo.opacity(0.2) : Color.secondary.opacity(0.08), in: RoundedRectangle(cornerRadius: 12))
                        }.buttonStyle(.plain)
                    }
                }
                if let outfit = store.outfit {
                    StudioCard(title: "Selected outfit") {
                        Text(outfit.guidance).foregroundStyle(.secondary)
                        HStack { TextField("Outfit name", text: $outfitName); Button("Rename") { store.renameOutfit(outfitName) }; Button("Set default") { store.defaultOutfit() }; Button("Replace photo…") { store.importPhoto(replace: true) }; Button("Delete", role: .destructive) { delete = true } }
                        Text("A seated body, desk, chair, and hands must already be present in your source photograph.").font(.caption).foregroundStyle(.secondary)
                    }.onAppear { outfitName = outfit.name }.onChange(of: store.selectedOutfit) { _, _ in outfitName = store.outfit?.name ?? "" }
                }
            }
        }.padding(28) }.confirmationDialog("Remove this outfit from the library? Original files and existing projects will be retained.", isPresented: $delete) { Button("Remove outfit", role: .destructive) { store.deleteOutfit() } }
            .sheet(isPresented: $chooseFrame) {
                if let source = frameVideo { VideoFrameImportView(source: source) { seconds in store.importVideoFrame(source: source, seconds: seconds) } }
            }
    }
}
struct PhotoView: View {
    let path: String
    @ViewState<CGImage?> private var image = nil
    @ViewState<Bool> private var loading = false
    var body: some View {
        Group {
            if let image { Image(decorative: image, scale: 1).resizable().scaledToFit() }
            else if loading { ProgressView("Loading photograph…") }
            else { ContentUnavailableView("No photograph selected", systemImage: "person.crop.rectangle", description: Text("Import a real outfit photo in Avatar Library.")) }
        }.task(id: path) {
            image = nil; loading = !path.isEmpty
            let source = path
            let result = await Task.detached(priority: .userInitiated) { PhotoPreview.load(path: source) }.value
            guard !Task.isCancelled else { return }
            image = result; loading = false
        }
    }
}
struct VideoStudioView: View {
    @EnvironmentObject var store: StudioStore
    @ViewState<Bool> private var safeAreas = false
    @ViewState<Double> private var zoom = 1.0
    @StateObject private var audioPlayback = AudioPlayback()
    var body: some View {
        HSplitView {
            ScrollView { VStack(alignment: .leading, spacing: 20) {
                Text("Compose your video").font(.title2.bold())
                TextField("Project name", text: $store.projectName)
                Picker("Avatar", selection: $store.selectedAvatar) { Text("Choose…").tag(""); ForEach(store.library.avatars) { Text($0.name).tag($0.id) } }.onChange(of: store.selectedAvatar) { _, id in store.selectedOutfit = store.library.avatars.first(where: {$0.id == id})?.defaultOutfit ?? "" }
                Picker("Outfit", selection: $store.selectedOutfit) { Text("Choose…").tag(""); ForEach(store.library.avatars.first(where: {$0.id == store.selectedAvatar})?.outfits ?? []) { Text($0.name).tag($0.id) } }
                Divider()
                Text("Voice-over").font(.headline)
                Button("Import audio…", systemImage: "waveform") { store.importAudio() }.disabled(store.busy)
                Text(store.audio.isEmpty ? "English, Hindi, or Hinglish. Use your recorded audio." : String(format: "Imported • %.1f seconds", store.audioDuration)).font(.caption).foregroundStyle(.secondary)
                Text("Script / production notes").font(.headline)
                TextEditor(text: $store.script).frame(height: 95).border(.quaternary)
                Text("Notes only. Speech comes from the imported recording.").font(.caption).foregroundStyle(.secondary)
                Divider()
                Picker("Background", selection: $store.backgroundMode) { ForEach(["Original", "Solid color", "Image"], id: \.self) { Text($0) } }
                if store.backgroundMode == "Solid color" { ColorPicker("Color", selection: Binding(get: {Color(nsColor: store.backgroundColor)}, set: {store.backgroundColor = NSColor($0)})) }
                if store.backgroundMode == "Image" { Button("Import background…") { if let url = store.pick([.image]) { store.backgroundPath = url.path } }; if !store.backgroundPath.isEmpty { Text(URL(fileURLWithPath: store.backgroundPath).lastPathComponent).font(.caption) } }
                if store.backgroundMode != "Original" { Text("Person segmentation may remove a desk or chair. Review the output edges.").font(.caption).foregroundStyle(.secondary) }
                Text("Video avatar").font(.headline)
                if store.renderEngine.usesAutomaticMotion {
                    Text("Uses your outfit photo and imported voice-over with automatic motion. Visual quality is still under review.").font(.caption).foregroundStyle(.secondary)
                    if !store.motion.isEmpty || !store.sourceVideo.isEmpty {
                        Button("Use photo and automatic motion") { store.motion = ""; store.sourceVideo = "" }
                    }
                    if store.renderEngine == .flashHead { Toggle("Reduce small head jitters", isOn: $store.stabilizeHead) }
                } else {
                    Picker("Visual source", selection: $store.sourceVideo) {
                        Text("Outfit photograph").tag("")
                        ForEach(store.library.motions, id: \.self) { Text(URL(fileURLWithPath: $0).lastPathComponent.prefix(20)).tag($0) }
                    }
                    Button("Import talking video…") { store.importMotion(); store.sourceVideo = store.motion; store.motion = "" }
                    if !store.sourceVideo.isEmpty {
                        Text("Real frames preserve your recorded clothing, background, blinking and movement. Source video must cover the full audio duration. Photo background controls do not affect this mode.").font(.caption).foregroundStyle(.secondary)
                    }
                    Picker("Motion", selection: $store.motion) { Text("Still pose • lip sync only").tag(""); ForEach(store.library.motions, id: \.self) { Text(URL(fileURLWithPath: $0).lastPathComponent.prefix(16)).tag($0) } }
                    Text("Reference motion requires local LivePortrait weights and runtime.").font(.caption).foregroundStyle(.secondary)
                }
                Picker("Format", selection: $store.portrait) { Text(store.renderEngine.formatLabel(portrait: false)).tag(false); Text(store.renderEngine.formatLabel(portrait: true)).tag(true) }
                Button("Add to render queue", systemImage: "play.fill") { store.enqueue() }.buttonStyle(.borderedProminent).controlSize(.large).disabled(store.outfit == nil || store.audio.isEmpty || store.busy)
                if store.renderEngine == .flashHead {
                    Button("Render test clip", systemImage: "film") { store.enqueue(testClip: true) }
                        .disabled(store.outfit == nil || store.audio.isEmpty || store.busy)
                    Text("Uses up to five seconds from the start of your voice-over. The full recording stays in your project.").font(.caption).foregroundStyle(.secondary)
                }
            }.padding(22) }.frame(minWidth: 280, idealWidth: 310, maxWidth: 360)
            VStack(spacing: 18) {
                HStack { Label(store.sourceVideo.isEmpty ? "Source photograph" : "Video avatar selected", systemImage: store.sourceVideo.isEmpty ? "photo" : "film").font(.headline); Spacer(); Toggle("Safe areas", isOn: $safeAreas).toggleStyle(.checkbox); Button("Fit") { zoom = 1 } }
                ZStack {
                    Color.black.opacity(0.7)
                    StudioPhotoPreview(request: StudioPhotoPreviewRequest(photo: store.outfit?.prepared ?? "",
                        backgroundMode: store.sourceVideo.isEmpty ? store.backgroundMode : "Original",
                        backgroundPath: store.backgroundPath, color: store.backgroundColor)).scaleEffect(zoom).padding(12)
                    if safeAreas { Rectangle().strokeBorder(.white.opacity(0.5), style: StrokeStyle(lineWidth: 1, dash: [7])).padding(40).allowsHitTesting(false) }
                }.aspectRatio(store.portrait ? 9/16 : 16/9, contentMode: .fit).clipped().clipShape(RoundedRectangle(cornerRadius: 12))
                HStack { Text(store.sourceVideo.isEmpty ? "Source and background preview • final framing is applied during rendering" : "Outfit thumbnail only • render uses the imported talking video").font(.caption).foregroundStyle(.secondary); Spacer(); Slider(value: $zoom, in: 1...2).frame(width: 90).accessibilityLabel("Preview zoom") }
                Spacer(minLength: 0)
                VStack(alignment: .leading, spacing: 12) {
                    HStack { Text("Voice-over timeline").font(.headline); Spacer(); Button("Play", systemImage: "play.fill") { audioPlayback.play() }.disabled(store.audio.isEmpty || audioPlayback.isPlaying); Button("Pause", systemImage: "pause.fill") { audioPlayback.pause() }.disabled(!audioPlayback.isPlaying) }
                    WaveformView(samples: store.waveform).frame(height: 64)
                    Slider(value: Binding(get: { audioPlayback.position }, set: { audioPlayback.seek(to: $0) }), in: 0...max(0.01, audioPlayback.duration)).disabled(store.audio.isEmpty).accessibilityLabel("Voice-over position")
                    HStack { Text(AudioPlayback.clock(audioPlayback.position)); Spacer(); Text(String(format: "%.1f s", store.audioDuration)) }.font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                    if let error = audioPlayback.error { Text(error).font(.caption).foregroundStyle(.red) }
                }.padding(16).background(.quaternary.opacity(0.3), in: RoundedRectangle(cornerRadius: 12))
            }.padding(24).frame(minWidth: 400)
            VStack(alignment: .leading, spacing: 20) {
                Text("Output inspector").font(.headline)
                Label("Local inference", systemImage: "desktopcomputer")
                Text("25 fps • H.264\nMOV with original recording\nBatch size 1").font(.callout).foregroundStyle(.secondary)
                Divider()
                Text("Appearance").font(.headline)
                Text("Clothing and body come from your photograph. Face animation can introduce artifacts; review every export.").foregroundStyle(.secondary)
                Divider()
                Text("Rendering").font(.headline)
                Text("One job at a time to limit memory pressure. Automatic idle sleep is prevented during rendering; the display can sleep. Progress and logs appear in Render Queue.").foregroundStyle(.secondary)
                Spacer()
            }.padding(20).frame(minWidth: 175, idealWidth: 200, maxWidth: 240)
        }.onAppear { audioPlayback.load(path: store.audio, duration: store.audioDuration) }
         .onChange(of: store.audio) { _, path in audioPlayback.load(path: path, duration: store.audioDuration) }
         .onDisappear { audioPlayback.pause() }
    }
}
struct WaveformView: View {
    let samples: [Float]
    var body: some View { Canvas { context, size in
        guard !samples.isEmpty else { return }
        let width = size.width / Double(samples.count)
        for (i, sample) in samples.enumerated() { let height = max(2, Double(sample) * size.height); context.fill(Path(roundedRect: CGRect(x: Double(i)*width, y: (size.height-height)/2, width: max(1, width-1), height: height), cornerRadius: 1), with: .color(.indigo)) }
    }.accessibilityLabel(samples.isEmpty ? "No audio imported" : "Waveform of imported voice-over") }
}
struct MotionView: View {
    @EnvironmentObject var store: StudioStore
    @StateObject private var camera = CameraCapture()
    var body: some View {
        ScrollView { VStack(alignment: .leading, spacing: 22) {
            Text("Capture your natural expression.").font(.title.bold())
            Text("Record 5–15 seconds of gentle nods, blinks, and a relaxed smile. Keep your face visible and camera steady. Recording stops at 30 seconds.").foregroundStyle(.secondary)
            HStack { Picker("Camera", selection: $camera.selected) { ForEach(camera.devices, id: \.uniqueID) { Text($0.localizedName).tag($0.uniqueID) } }.disabled(camera.recording || camera.preparing || camera.recordingRequested || camera.countdown > 0); Button(camera.preparing ? "Starting preview…" : "Start preview") { camera.start() }.disabled(camera.recording || camera.preparing || camera.recordingRequested || camera.countdown > 0); Button("Import reference video…") { store.importMotion() } }
            ZStack { Color.black; CameraPreview(session: camera.session); if camera.countdown > 0 { Text("\(camera.countdown)").font(.system(size: 80, weight: .bold)).foregroundStyle(.white) } }.frame(height: 380).clipShape(RoundedRectangle(cornerRadius: 14))
            HStack { Button(camera.recording ? "Stop recording" : "Record with microphone", systemImage: camera.recording ? "stop.circle.fill" : "record.circle") { if camera.recording { camera.stopRecording() } else { camera.record() } }.buttonStyle(.borderedProminent).disabled(!camera.ready || camera.countdown > 0 || camera.recordingRequested); Text("\(camera.elapsed) s").monospacedDigit(); if let error = camera.error { Text(error).foregroundStyle(.red) } }
            Text("Saved motion references").font(.headline)
            ForEach(store.library.motions, id: \.self) { path in HStack { Label(URL(fileURLWithPath: path).lastPathComponent, systemImage: "film"); Spacer(); Button("Preview") { NSWorkspace.shared.open(URL(fileURLWithPath: path)) }; Button("Use reference") { store.motion = path } } }
        }.padding(28) }.onAppear { camera.recordingDirectory = store.root.appendingPathComponent("recordings"); camera.discover(); camera.onSaved = { store.storeMotion($0) } }.onDisappear { camera.stop() }
    }
}
struct JobsView: View {
    @EnvironmentObject var store: StudioStore
    let projects: Bool
    @ViewState<AVPlayer?> private var player: AVPlayer?
    var body: some View {
        ScrollView { VStack(alignment: .leading, spacing: 18) {
            if let player { VideoPlayer(player: player).frame(height: 320) }
            if store.library.jobs.isEmpty { ContentUnavailableView("No videos yet", systemImage: "film.stack", description: Text("Choose a photo and voice-over in Create Video.")) }
            ForEach(store.library.jobs.reversed()) { job in
                StudioCard(title: job.name) {
                    HStack { Text(job.status).font(.headline).foregroundStyle(job.status == "Failed" ? .red : .primary); Spacer(); Text(job.created, style: .date).foregroundStyle(.secondary) }
                    ProgressView(value: job.progress)
                    Text(job.detail).font(.callout).textSelection(.enabled)
                    if job.status == "Completed" {
                        Label(job.review?.decision.label ?? "Rendered • visual review pending", systemImage: job.review?.decision == .looksNatural ? "checkmark.bubble" : "eye")
                            .font(.callout).foregroundStyle(.secondary)
                        HStack {
                            Button("Looks natural") { store.review(job, decision: .looksNatural) }
                            Button("Needs refinement") { store.review(job, decision: .needsRefinement) }
                            if job.review != nil { Button("Clear review") { store.review(job, decision: nil) } }
                        }
                        Text("Review head stability, eyes and lip sync during playback with sound.").font(.caption).foregroundStyle(.secondary)
                    }
                    HStack {
                        if job.status == "Completed" { Button("Preview", systemImage: "play") { player = AVPlayer(url: URL(fileURLWithPath: job.output)); player?.play() }; Button("Export…", systemImage: "square.and.arrow.up") { store.export(job) } }
                        if ["Queued", "Rendering"].contains(job.status) { Button("Cancel", role: .destructive) { store.cancel(job) } }
                        if ["Failed", "Interrupted", "Cancelled"].contains(job.status) {
                            Button((job.engine ?? .museTalk).supportsResume ? "Resume" : "Retry") { store.retry(job) }
                            if (job.engine ?? .museTalk).supportsResume { Button("New render") { store.retry(job, resume: false) } }
                        }
                        Button("Open log") { store.revealLog(job) }
                    }
                }
            }
        }.padding(28) }.onDisappear { player?.pause() }
    }
}
struct SettingsView: View {
    @EnvironmentObject var store: StudioStore
    @AppStorage("appearance") var appearance = "Dark"
    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            Text("Local runtime & preferences").font(.title2.bold())
            Picker("Appearance", selection: $appearance) { ForEach(["Dark", "Light", "System"], id: \.self) { Text($0) } }.frame(width: 280)
            Picker("Video engine", selection: $store.renderEngine) {
                ForEach(RenderEngine.allCases, id: \.self) { Text($0.label).tag($0) }
            }.frame(maxWidth: 440)
            if store.renderEngine == .ditto {
                Text("Ditto evaluation uses the local model files and voice-overs up to 15 seconds. Run diagnostics to check setup.").font(.caption).foregroundStyle(.secondary)
            } else if store.renderEngine != .museTalk { GroupBox(store.renderEngine == .flashHead ? "FlashHead local generation setup" : "LongCat local generation setup") {
                VStack(alignment: .leading, spacing: 8) {
                    ProgressView(value: store.modelStatus.fraction)
                    Text(store.modelStatus.detail).font(.callout)
                    Text(store.modelStatus.evaluation).font(.caption).foregroundStyle(.secondary)
                    Button("Open evaluation log") {
                        NSWorkspace.shared.open(URL(fileURLWithPath: store.runtime).appendingPathComponent(store.modelStatus.evaluationLog))
                    }
                }.padding(6)
            } }
            TextField("Local runtime folder", text: $store.runtime)
            HStack { Button("Run diagnostics") { store.diagnose() }; Button("Open local library") { NSWorkspace.shared.open(store.root) } }
            ScrollView { Text(store.diagnostics).font(.system(.caption, design: .monospaced)).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading).padding(14) }.background(.quaternary.opacity(0.3), in: RoundedRectangle(cornerRadius: 10))
            Text("Generation stays on this Mac. Import your recorded voice; personalized model training and voice cloning are not implemented. Run diagnostics to check the selected engine's installation.").font(.caption).foregroundStyle(.secondary)
        }.preferredColorScheme(appearance == "System" ? nil : appearance == "Dark" ? .dark : .light)
    }
}
