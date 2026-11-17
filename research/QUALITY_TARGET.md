# Acceptance criteria for realistic avatar output

The user's target is HeyGen-level realism. Runtime success is necessary but not sufficient. Current app outputs have been rejected visually.

For each candidate, retain source footage, exact driving audio, seed/settings, model revision, device/memory/time, source/candidate close-ups, and a playable candidate. Compare the same short segment before committing to full-length processing.

Release criteria:
- No obvious moustache, lip, teeth, skin-tone, or chin identity changes.
- No visible mouth patch, ghost edges, unstable jaw, shimmer, or flicker during normal playback and close-up.
- Phoneme timing credible on new speech, including plosive closures and pauses; inspect English/Hindi/Hinglish separately.
- Eyes and body natural; record when gestures come from source video rather than synthesizing claims.
- Original audio preserved; objective decode equality checked when formats permit.
- No unexplained cut, reversed source movement, repeated frames, or audio drift.
- User review against a specific quality reference, when provided.

These criteria are not a guarantee that any available local model meets them. Scores alone cannot approve a candidate. A sharper source capture may help but cannot repair a model's reconstruction ceiling.

Requested resources: optional personal HeyGen-quality reference, and optional sharper stable 1080p/4K talking footage with even light and no beauty filter. Continue using current files while awaiting these resources. No cloud inference authorization has been requested or inferred.
