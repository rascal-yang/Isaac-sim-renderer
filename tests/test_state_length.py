"""Recorder schema must agree with allocated rows for a non-demo trajectory."""
import importlib.util,sys,tempfile,unittest,types,os
from pathlib import Path
from unittest.mock import patch
import numpy as np
class Length(unittest.TestCase):
    def test_non_demo_recorder_schema(self):
        root=Path(__file__).resolve().parents[1]
        spec=importlib.util.spec_from_file_location('recorder',root/'integrations/g1/state_stream.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        value=types.SimpleNamespace();value.detach=lambda:value;value.cpu=lambda:value;value.numpy=lambda:np.zeros((1,1),np.float32)
        data=types.SimpleNamespace(**{name:value for name in m.FIELDS+m.JOINT_FIELDS})
        asset=types.SimpleNamespace(data=data,cfg=types.SimpleNamespace(prim_path='/World/Robot'),_physics_sim_view=object())
        scene=types.SimpleNamespace(articulations={'robot':asset},rigid_objects={},deformable_objects={},rigid_object_collections={},update=lambda **kw:None)
        env=types.SimpleNamespace(scene=scene)
        usd=types.ModuleType('omni.usd');usd.get_context=lambda:types.SimpleNamespace(get_stage=lambda:types.SimpleNamespace(Traverse=lambda:[]))
        omni=types.ModuleType('omni');omni.usd=usd
        pxr=types.ModuleType('pxr');pxr.UsdPhysics=object();pxr.UsdGeom=object()
        with tempfile.TemporaryDirectory() as d,patch.dict(sys.modules,{'omni':omni,'omni.usd':usd,'pxr':pxr}),patch.dict(os.environ,{'G1_MAX_STEPS':'3'}):
            recorder=m.StateRecorder(types.SimpleNamespace(env=env,out=d,active=False))
            self.assertEqual(recorder.frames.shape,(4,))
            self.assertEqual(recorder.physics.shape,(13,))
            self.assertEqual(recorder.schema['frame_rows'],len(recorder.frames))
            self.assertEqual(recorder.schema['physics_rows'],len(recorder.physics))
