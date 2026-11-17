"""Mathematical compatibility checks; synthetic tensors are not avatar results."""
import unittest
import torch
from flashhead_mps import load_model_module,rotary,frequencies_3d,attention,sinusoidal_embedding


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=load_model_module()

    def test_real_rotary_matches_official_complex_cpu(self):
        generator=torch.Generator().manual_seed(16)
        for grid in [(1,3,4),(3,4,5)]:
            values=torch.randn(1,grid[0]*grid[1]*grid[2],3,128,generator=generator)
            expected=self.model.reference_rotary(values,self.model.reference_frequencies(128),grid)
            actual=rotary(values,frequencies_3d(128),grid)
            torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)

    def test_bounded_attention_matches_whole_query(self):
        generator=torch.Generator().manual_seed(27)
        q=torch.randn(1,37,96,generator=generator)
        k=torch.randn(1,51,96,generator=generator);v=torch.randn(1,51,96,generator=generator)
        actual=attention(q,k,v,3,query_chunk=7)
        expected=attention(q,k,v,3,query_chunk=100)
        torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)

    def test_embedding_matches_official_double_cpu(self):
        position=torch.tensor([0,250,500,750,1000],dtype=torch.float32)
        torch.testing.assert_close(sinusoidal_embedding(256,position),
                                   self.model.reference_sinusoidal(256,position),atol=0,rtol=0)

    @unittest.skipUnless(torch.backends.mps.is_available(),'Requires actual Metal access')
    def test_mps_bfloat_timestep_matches_official_cpu(self):
        position=torch.tensor([0,250,500,750,1000],dtype=torch.bfloat16,device='mps')
        actual=sinusoidal_embedding(256,position).cpu()
        expected=self.model.reference_sinusoidal(256,position.cpu())
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)

    def test_distributed_or_wrong_geometry_refused(self):
        x=torch.zeros(1,12,3,128);freqs=frequencies_3d(128)
        with self.assertRaises(RuntimeError):rotary(x,freqs,(1,3,4),use_usp=True)
        with self.assertRaises(ValueError):rotary(x,freqs,(1,3,3))


if __name__=='__main__':unittest.main()
