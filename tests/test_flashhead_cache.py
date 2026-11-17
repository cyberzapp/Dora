import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
from flashhead_cache import ChunkCache


class CacheTests(unittest.TestCase):
    shape=(28,8,8,3)

    def test_interrupted_commit_is_not_a_completed_chunk(self):
        with tempfile.TemporaryDirectory() as folder:
            frames=np.zeros(self.shape,dtype=np.uint8)
            with ChunkCache(folder,{'source':'A'},self.shape) as cache:
                # Simulate power loss after the RGB/state writes but before commit.
                with patch('flashhead_cache.atomic_json',side_effect=OSError('disk interrupted')):
                    with self.assertRaises(OSError):cache.commit(0,frames,b'opaque test state',28)
            with ChunkCache(folder,{'source':'A'},self.shape) as cache:
                self.assertEqual(cache.records,[])
                cache.commit(0,frames,b'opaque test state',28)
            with ChunkCache(folder,{'source':'A'},self.shape) as cache:self.assertEqual(len(cache.records),1)

    def test_changed_source_and_corrupted_state_are_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            with ChunkCache(folder,{'source':'A'},self.shape) as cache:
                cache.commit(0,np.zeros(self.shape,dtype=np.uint8),b'opaque test state',28)
            with self.assertRaisesRegex(ValueError,'settings changed'):ChunkCache(folder,{'source':'B'},self.shape)
            (Path(folder)/'chunk-0000.state').write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError,'verification'):ChunkCache(folder,{'source':'A'},self.shape)

    def test_parallel_writer_and_missing_commit_are_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            with ChunkCache(folder,{'source':'A'},self.shape) as cache:
                with self.assertRaises(BlockingIOError):ChunkCache(folder,{'source':'A'},self.shape)
                with self.assertRaisesRegex(ValueError,'contiguous'):cache.commit(1,np.zeros(self.shape,dtype=np.uint8),b'state',28)

    def test_frame_corruption_and_commit_gap_are_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            with ChunkCache(folder,{'source':'A'},self.shape) as cache:
                for index in range(2):cache.commit(index,np.zeros(self.shape,dtype=np.uint8),b'state',28)
            path=Path(folder)/'chunk-0001.npy'
            data=bytearray(path.read_bytes());data[-1]^=1;path.write_bytes(data)
            with self.assertRaisesRegex(ValueError,'verification'):ChunkCache(folder,{'source':'A'},self.shape)
            (Path(folder)/'commit-0000.json').unlink()
            with self.assertRaisesRegex(ValueError,'gap'):ChunkCache(folder,{'source':'A'},self.shape)


if __name__=='__main__':unittest.main()
