"""Isolated fixed-token replay with task-preserving or official-like XR settings."""
import argparse
import json
import os
import sys
import fcntl
# Choose RT2 before Kit startup, matching the previously tested native-DLAA profile.
from isaaclab.app import AppLauncher
_app_init = AppLauncher.__init__
def _rt2_app_init(self, launcher_args, **kwargs):
    launcher_args.renderer = 'RealTimePathTracing'
    launcher_args.kit_args += ' --/persistent/rtx/modes/rt2/enabled=true --/app/renderer/waitIdle=true --/renderer/multiGpu/enabled=false --/renderer/activeGpu=0'
    launcher_args.kit_args += ' --/persistent/physics/numThreads=8'
    if os.environ.get('PIPE_AFFINITY'):
        launcher_args.kit_args += ' --/plugins/carb.tasking.plugin/threadCount=8'
    return _app_init(self, launcher_args, **kwargs)
AppLauncher.__init__ = _rt2_app_init
sys.path.insert(0, str(__import__('pathlib').Path(os.environ['RENDER_G1_RUNTIME'])/'workspace/pipelines/sonic_teleop_augmentation'))
from common import BASE, HERE, MODEL, W, arrays, module, read, sha

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['replay'], default='replay')
parser.add_argument('--candidate', type=__import__('pathlib').Path, required=True)
parser.add_argument('--output', type=__import__('pathlib').Path, required=True)
parser.add_argument('--benchmark-reference', type=__import__('pathlib').Path)
parser.add_argument('--variant', choices=['sonic50'], default='sonic50')
parser.add_argument('--runtime', choices=['fast'], default='fast')
args = parser.parse_args()
os.environ['FAST_RUNTIME_GENERATION'] = args.runtime
os.environ['XR_REPLAY_VARIANT'] = args.variant
os.environ.setdefault('PROFILE_CAMERA', '1')
os.environ.setdefault('PROFILE_DETAIL', '1')
gpu_lock = (W/'experiments/.sonic_augmentation_gpu.lock').open('a')
if os.environ.get('PIPE_COORDINATED')!='1':
    fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
candidate, out = args.candidate.resolve(), args.output.resolve()
assert not out.exists(), out
for name, digest in read(candidate/'prepared_files.json').items():
    assert sha(candidate/name) == digest, name
token_times = arrays(candidate/'grasp_tokens.npz')['time_s']
expected_dt = .05 if args.variant == 'physics200' else .02
assert __import__('numpy').allclose(__import__('numpy').diff(token_times), expected_dt, atol=1e-5), 'Candidate time grid does not match variant; use candidate_20hz only with official'
initial = arrays(candidate/'initial_state.npz')
maximum = 1800 if args.mode == 'generate' else len(arrays(candidate/'grasp_tokens.npz')['time_s'])
os.environ.update(G1_LEAN_MODE=args.mode, G1_AUG_CANDIDATE=str(candidate),
    G1_START_ROOT=json.dumps(initial['root_position'].tolist()), G1_HEADLESS='1',
    G1_OUTPUT_DIR=str(out), G1_GROUP='window_scene', G1_ROOM='window_room',
    G1_DESK_PLUSH='1', GRASP_METHOD='feedback', GRASP_CONTROLLER='sonic_full',
    G1_AUTO_CAPTURE='1', G1_LIVE_SESSION='1', G1_MAX_STEPS=str(maximum),
    G1_GRAIL_ACTIVE='0', G1_REVO2='1', G1_STATIONARY='0', G1_OPTIMIZED='1')
os.environ.pop('G1_LEAN_STOP_REASON', None)
if args.benchmark_reference:
    os.environ['G1_LEAN_BENCHMARK_REFERENCE'] = str(args.benchmark_reference.resolve())
else:
    os.environ.pop('G1_LEAN_BENCHMARK_REFERENCE', None)
sys.path.insert(0, str(BASE/'code'))
sys.path.insert(0, str(HERE))
sys.modules['training_capture'] = module('xr_training_capture', __import__('pathlib').Path(__file__).parent/'xr_capture.py')
sys.modules['contact_watch_full'] = module('contact_watch_full', BASE/'code/contact_watch.py')
sys.modules['contact_watch'] = module('xr_contact_watch', __import__('pathlib').Path(__file__).parent/'xr_contact_watch.py')
sys.modules['runtime_adapter'] = module('lean_runtime_adapter', __import__('pathlib').Path(__file__).parent/'lean_runtime_adapter.py')
if args.mode == 'generate':
    from mobile_supervisor import install as policy
    from mobile_controller import install as controller
    policy()
    controller()
else:
    if (candidate/'provenance.json').exists():
        assert sha(MODEL/'model_decoder.onnx') == read(candidate/'provenance.json')['decoder_sha256']
    from automatic_policy import install as policy
    from replay_controller import install as controller
    policy()
    controller(candidate/'grasp_tokens.npz', MODEL/'model_decoder.onnx',
               candidate/'deployment_parameters.json', W/'g1_closed_loop/.deps')
from xr_setup import install as install_xr
install_xr()
from fast_runtime import install as install_fast
install_fast()
from instrumentation import install as install_profile
install_profile()
from pipeline_hook import install as install_pipeline
install_pipeline()
from experiment import run
run(out.name, sonic_planner=True, seed=17)
