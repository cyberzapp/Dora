import Foundation

@main struct RenderEngineMigrationSmoke {
    static func main() throws {
        var job = RenderJob(name: "Saved project", photo: "photo", audio: "audio", motion: "", script: "", sourceVideo: nil, portrait: false)
        let encoder = JSONEncoder(), decoder = JSONDecoder()
        var legacy = try JSONSerialization.jsonObject(with: encoder.encode(job)) as! [String: Any]
        legacy.removeValue(forKey: "engine")
        let restored = try decoder.decode(RenderJob.self, from: JSONSerialization.data(withJSONObject: legacy))
        precondition(restored.engine == nil, "Legacy jobs must retain the legacy route")
        job.engine = .longCat
        job.faceBounds = [0.3, 0.4, 0.2, 0.25]
        let longCat = try decoder.decode(RenderJob.self, from: encoder.encode(job))
        precondition(longCat.engine == .longCat)
        precondition(longCat.faceBounds == job.faceBounds)
        precondition(longCat.id == job.id && longCat.audio == job.audio)
        job.engine = .flashHead; job.stabilizeHead = true
        let flashHead = try decoder.decode(RenderJob.self, from: encoder.encode(job))
        precondition(flashHead.engine == .flashHead && flashHead.stabilizeHead == true)
        precondition(restored.stabilizeHead == nil, "Legacy jobs must decode without the new stabilization field")
        print("Legacy project decoding and saved LongCat/FlashHead engines: passed")
    }
}
