import collections
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
from render_storage import require_render_space


class StorageRecoveryTests(unittest.TestCase):
    def test_low_disk_fails_before_render_and_allows_retry_after_space_is_freed(self):
        usage = collections.namedtuple('usage', 'total used free')
        with patch('render_storage.shutil.disk_usage', return_value=usage(100, 99, 1)):
            with self.assertRaisesRegex(OSError, 'Free space and resume this job'):
                require_render_space('.', 1328, 768, 432)
        with patch('render_storage.shutil.disk_usage', return_value=usage(100*1024**3, 0, 100*1024**3)):
            needed = require_render_space('.', 1328, 768, 432)
            self.assertGreater(needed, 1328*768*432*3*2)


if __name__ == '__main__': unittest.main()
