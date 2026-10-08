import importlib.util
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('delivery', ROOT/'launch.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class PortableTests(unittest.TestCase):
    def test_frame_counts(self):
        self.assertEqual(m.state_sizes(3000), (12001,3001))
        with self.assertRaises(ValueError): m.state_sizes(0)
    def test_cpu_groups_do_not_share_cores(self):
        pairs=[[0,8],[1,9],[2,10],[3,11]]
        a,b=m.split_cores(pairs)
        self.assertTrue(a and b)
        self.assertFalse(set(a)&set(b))
        for pair in pairs: self.assertTrue(set(pair)<=set(a) or set(pair)<=set(b))
    def test_missing_assets_actionable(self):
        with self.assertRaisesRegex(FileNotFoundError,'交付资源'): m.resource_package(Path('/nonexistent'))
    def test_missing_reference_file_fails(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);src=root/'src';ref=root/'ref';src.mkdir();ref.mkdir()
            (src/'full_physics_states.npy').write_bytes(b'example')
            (ref/'full_physics_states.npy').write_bytes(b'example')
            with self.assertRaises(FileNotFoundError):m.compare_physical(src,ref)
if __name__=='__main__':unittest.main()
