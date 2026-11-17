import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
from flashhead_timeline import plan,NEW_FRAMES,NEW_SAMPLES


class TimelineTests(unittest.TestCase):
    def test_actual_full_voice_keeps_final_fractional_frame(self):
        frames,windows=plan(849749) # supplied voice: 53.1093125 seconds
        self.assertEqual(frames,1328)
        self.assertEqual(len(windows),48)
        self.assertEqual(windows[-1].append_frames,12)
        self.assertEqual(windows[-1].analysis_end,849749)
        self.assertEqual(sum(w.append_frames for w in windows),1328)

    def test_generated_output_and_analysis_have_no_gaps_or_repeats(self):
        for samples in [8000,17920,17921,77440,849749,4800000]:
            frames,windows=plan(samples)
            self.assertEqual([i for w in windows for i in range(w.start_frame,w.start_frame+w.append_frames)],list(range(frames)))
            self.assertEqual(windows[0].analysis_start,0)
            for left,right in zip(windows,windows[1:]):self.assertEqual(left.analysis_end,right.analysis_start)
            self.assertEqual(windows[-1].analysis_end,samples)
            for window in windows:
                self.assertLessEqual(window.append_frames,NEW_FRAMES)
                self.assertEqual(window.analysis_end-window.analysis_start+window.analysis_pad_after,NEW_SAMPLES)

    def test_five_second_benchmark_geometry_is_preserved(self):
        frames,windows=plan(77440) # real 4.84-second benchmark
        self.assertEqual((frames,len(windows),windows[-1].append_frames),(121,5,9))

    def test_native_audio_fraction_is_not_lost_during_resampling(self):
        # A tiny native-rate tail may round down to 1.12 s at 16 kHz. The video
        # must still cover the original audio, which extends into frame 29.
        frames,windows=plan(17920,output_frames=29)
        self.assertEqual((frames,len(windows),windows[-1].append_frames),(29,2,1))
        self.assertEqual(windows[-1].analysis_pad_after,NEW_SAMPLES)
        with self.assertRaises(ValueError):plan(17920,output_frames=31)


if __name__=='__main__':unittest.main()
