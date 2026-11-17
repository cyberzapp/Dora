"""Exercise real encoded timing failures, rather than mocked decoder output."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import imageio_ffmpeg

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from verify_export import verify_video


class ExportTimestampTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='avatar-cadence-test-')
        cls.folder = Path(cls.temporary.name)
        for name, filter_ in (
            ('regular', 'settb=1/1000'),
            ('irregular', "settb=1/1000,setpts='PTS+if(eq(N,10),10,0)'"),
        ):
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-f', 'lavfi',
                '-i', 'testsrc2=s=64x64:r=25:d=0.8', '-vf', filter_, '-enc_time_base', '1:1000',
                '-fps_mode', 'vfr', '-c:v', 'libx264', '-video_track_timescale', '1000',
                str(cls.folder / f'{name}.mov')], check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_regular_export(self):
        self.assertEqual(verify_video(self.folder / 'regular.mov', 20, 25), 20)

    def test_twenty_decodable_frames_can_still_have_irregular_timing(self):
        path = self.folder / 'irregular.mov'
        self.assertEqual(verify_video(path, 20), 20)
        with self.assertRaisesRegex(RuntimeError, 'Irregular video presentation time at frame 10'):
            verify_video(path, 20, 25)

    def test_incomplete_export_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'Incomplete video'):
            verify_video(self.folder / 'regular.mov', 21, 25)


if __name__ == '__main__':
    unittest.main()
