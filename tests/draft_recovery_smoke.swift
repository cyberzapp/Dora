import Foundation
import AppKit

@main struct DraftRecoverySmoke {
    @MainActor static func main() async throws {
        let root = URL(fileURLWithPath: CommandLine.arguments[1]).appendingPathComponent(UUID().uuidString)
        let suite = "local.avatarstudio.draft-test.\(UUID().uuidString)"
        let preferences = UserDefaults(suiteName: suite)!
        defer { preferences.removePersistentDomain(forName: suite); try? FileManager.default.removeItem(at: root) }
        let first = StudioStore(storageRoot: root, preferences: preferences)
        let media = root.appendingPathComponent("audio/fixture.wav")
        try Data([1, 2, 3]).write(to: media) // Persistence fixture only; never passed to a renderer.
        let outfit = Outfit(name: "Original outfit", original: "photo.png", prepared: "prepared.png", guidance: "")
        let avatar = Avatar(name: "Consenting fixture", consent: true, outfits: [outfit], defaultOutfit: outfit.id)
        first.library.avatars = [avatar]; first.selectedAvatar = avatar.id; first.selectedOutfit = outfit.id
        first.projectName = "A saved educational video"; first.script = "Private script — नमस्ते"
        first.audio = media.path; first.audioDuration = 53.1093125; first.waveform = [0.1, 0.7, 0.2]
        first.renderEngine = .longCat; first.portrait = true
        first.backgroundMode = "Solid color"
        first.backgroundColor = NSColor(srgbRed: 0.2, green: 0.4, blue: 0.6, alpha: 1)
        // Exercise periodic autosave without explicitly calling save or shutting down.
        try await Task.sleep(for: .milliseconds(800))
        let snapshot = try Database(url: root.appendingPathComponent("studio.sqlite")).load()
        precondition(snapshot.draft?.name == first.projectName, "Timer failed to save edits")
        first.shutdown()
        let second = StudioStore(storageRoot: root, preferences: preferences)
        precondition(second.selectedAvatar == avatar.id && second.selectedOutfit == outfit.id)
        precondition(second.audio == media.path && second.audioDuration == 53.1093125)
        precondition(second.waveform == [0.1, 0.7, 0.2] && second.script == first.script)
        precondition(second.renderEngine == .longCat && second.portrait)
        precondition(second.backgroundMode == "Solid color")
        let color = second.backgroundColor.usingColorSpace(.sRGB)!
        precondition(abs(color.redComponent - 0.2) < 0.0001)
        second.shutdown()
        try FileManager.default.removeItem(at: media)
        let third = StudioStore(storageRoot: root, preferences: preferences)
        precondition(third.audio.isEmpty && third.waveform.isEmpty && third.audioDuration == 0)
        precondition(third.error?.contains("voice-over") == true, "Missing file guidance lost")
        precondition(third.projectName == first.projectName && third.script == first.script)
        third.shutdown()
        let legacy = Data("{\"avatars\":[],\"jobs\":[],\"motions\":[]}".utf8)
        let decoded = try JSONDecoder().decode(Library.self, from: legacy)
        precondition(decoded.draft == nil)
        print("Draft autosave, restart restoration, missing-file recovery and legacy migration: passed")
    }
}
