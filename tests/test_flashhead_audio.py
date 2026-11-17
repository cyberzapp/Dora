import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
from flashhead_audio import history_for
from flashhead_timeline import NEW_SAMPLES,HISTORY_SAMPLES


class AudioHistoryTests(unittest.TestCase):
    def test_direct_histories_match_the_tested_rolling_stream(self):
        speech=np.random.default_rng(73).standard_normal(849749).astype(np.float32)
        count=(len(speech)+NEW_SAMPLES-1)//NEW_SAMPLES
        padded=np.pad(speech,(0,count*NEW_SAMPLES-len(speech)))
        rolling=np.zeros(HISTORY_SAMPLES,dtype=np.float32)
        for index in range(count):
            rolling=np.concatenate((rolling[NEW_SAMPLES:],padded[index*NEW_SAMPLES:(index+1)*NEW_SAMPLES]))
            np.testing.assert_array_equal(history_for(speech,index),rolling)


if __name__=='__main__':unittest.main()
