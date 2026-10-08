"""Experiment-local compatibility layer; never patches files in the original pipeline."""
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT/'vendor/haptic_feedback.py'


def install():
    import runtime_adapter
    previous_install = runtime_adapter.install

    def installation(evaluation, out, intern):
        previous_install(evaluation, out, intern)
        previous_create = evaluation.train_agent_trl.create_manager_env

        def create(*args, **kwargs):
            import carb
            from isaaclab.envs import ManagerBasedRLEnv
            import brainco_adapter
            previous_configure = brainco_adapter.configure_scene

            def configure(scene):
                previous_configure(scene)
                # Same body coverage, combine object/table/room filters for each body.
                # Keep one body per Isaac Lab 2.3 filtered sensor.
                for name in list(vars(scene)):
                    sensor = getattr(scene, name)
                    if name.startswith('obstacle_watch_'):
                        setattr(scene, name, None)
                    elif name.startswith(('grasp_finger_', 'table_watch_')):
                        sensor.filter_prim_paths_expr = list(sensor.filter_prim_paths_expr)
                        if name.startswith('table_watch_left_') and name.removeprefix('table_watch_left_').isdigit():
                            sensor.filter_prim_paths_expr.append('{ENV_REGEX_NS}/Object')
                        sensor.filter_prim_paths_expr.append('{ENV_REGEX_NS}/Room')
                        sensor.track_pose = False
                        sensor.track_contact_points = False
                        sensor.track_friction_forces = False
                        # Full 200 Hz history is stored by ContactWatch. No consumer
                        # of these merged sensors reads the GPU history buffers.
                        sensor.history_length = 0
            brainco_adapter.configure_scene = configure
            previous_init = ManagerBasedRLEnv.__init__

            def env_init(env, cfg, *a, **kw):
                official = True
                physics200 = True
                if official:
                    for name in ['robot','brainco_left','brainco_right']:
                        asset=getattr(cfg.scene,name)
                        asset.spawn.rigid_props.disable_gravity=False
                        asset.spawn.articulation_props.solver_position_iteration_count=8 if name == 'robot' else 32
                        asset.spawn.articulation_props.solver_velocity_iteration_count=4 if name == 'robot' else 8
                if os.environ.get('PHYSX_TEST_MODE')=='cpu':
                    cfg.sim.device='cpu'
                cfg.sim.gravity = (0., 0., -9.81)
                for asset in vars(cfg.scene).values():
                    props = getattr(getattr(asset, 'spawn', None), 'rigid_props', None)
                    if props is not None:
                        props.disable_gravity = False
                cfg.sim.dt = .005 if physics200 else 1/120
                cfg.decimation = 4 if physics200 else 2
                cfg.sim.render_interval = 1 if physics200 else 2
                cfg.sim.render.antialiasing_mode = 'DLAA'
                settings = carb.settings.get_settings()
                for key, value in {
                    '/exts/isaacsim.core.throttling/enable_async': False,
                    '/app/asyncRendering': False,
                    '/app/asyncRenderingLowLatency': False,
                    '/omni/replicator/asyncRendering': False,
                    '/app/useFabricSceneDelegate': True,
                    '/rtx/hydra/readTransformsFromFabricInRenderDelegate': True,
                    '/xr/skipInputDeviceUSDWrites': True,
                    '/persistent/xr/profile/ar/renderQuality': 'performance',
                }.items():
                    settings.set(key, value)
                if not physics200:
                    inherited_step_dt = ManagerBasedRLEnv.step_dt.fget
                    ManagerBasedRLEnv.step_dt = property(lambda self: .02 if getattr(self, '_sonic50_schedule', False) else inherited_step_dt(self))
                    env._sonic50_schedule = True
                previous_init(env, cfg, *a, **kw)
                if not physics200:
                    old_step = env.step
                    schedule = dict(index=0)
                    def scheduled_step(action):
                        i = schedule['index']
                        env.cfg.decimation = round((i+1)*2.4)-round(i*2.4)
                        schedule['index'] += 1
                        return old_step(action)
                    env.step = scheduled_step
                # Headless capture now explicitly renders once per post-physics state.
                # Disable automatic substep renders; the physical schedule is unchanged.
                settings.set('/app/asyncRendering', False)
                settings.set('/app/asyncRenderingLowLatency', False)
                from pxr import UsdPhysics, PhysxSchema
                import omni.usd
                stage = omni.usd.get_context().get_stage()
                bodies = []
                for prim in stage.Traverse():
                    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                        disabled = PhysxSchema.PhysxRigidBodyAPI(prim).GetDisableGravityAttr().Get()
                        bodies.append(dict(path=str(prim.GetPath()), disable_gravity=disabled))
                (out/'all_gravity_audit.json').write_text(json.dumps(dict(gravity=list(cfg.sim.gravity), bodies=bodies, disabled_count=sum(x['disable_gravity'] is True for x in bodies)), indent=2))
                assert not any(x['disable_gravity'] is True for x in bodies), 'A spawned rigid body still disables gravity'
                # Render using the environment's substep schedule even in headless mode.
                env.sim.has_rtx_sensors = lambda: False
                report = dict(sonic_scheduler='fixed 4 physical steps, exact50Hz' if physics200 else 'nearest physical tick: alternating 2/3 steps, average50Hz', effective_render_hz=50, variant=os.environ['XR_REPLAY_VARIANT'], physics_dt=env.physics_dt,
                    control_dt=env.step_dt, decimation=cfg.decimation, render_interval=cfg.sim.render_interval,
                    device=env.device, all_sensor_instances=len(env.scene.sensors),
                    contact_sensor_instances=sum(isinstance(v,__import__('isaaclab.sensors',fromlist=['ContactSensor']).ContactSensor) for v in env.scene.sensors.values()),
                    collision_geometry_unchanged=True, cloudxr_connected=False, hardware_haptic_output=False,
                    haptic_source=str(UPSTREAM), compatibility='Official force aggregation/driver with 2.3 tensor adapters',
                    rendering='RT2/native DLAA, manual one update per post-physics state at50Hz; RGB retained; no headset/network',
                    settings={key: settings.get(key) for key in ['/app/asyncRendering','/app/asyncRenderingLowLatency',
                        '/omni/replicator/asyncRendering','/app/useFabricSceneDelegate','/rtx/post/aa/op']},
                    robot_properties={name:dict(disable_gravity=getattr(cfg.scene,name).spawn.rigid_props.disable_gravity,
                        position_iterations=getattr(cfg.scene,name).spawn.articulation_props.solver_position_iteration_count,
                        velocity_iterations=getattr(cfg.scene,name).spawn.articulation_props.solver_velocity_iteration_count)
                        for name in ['robot','brainco_left','brainco_right']})
                (out/'xr_runtime_config.json').write_text(json.dumps(report, indent=2))
            ManagerBasedRLEnv.__init__ = env_init
            wrapped=previous_create(*args, **kwargs)
            token_file=out/'token_controller.json'
            token_report=json.loads(token_file.read_text())
            token_report.update(frequency_Hz=1/wrapped.env.step_dt, physics_frequency_Hz=1/wrapped.env.physics_dt,
                temporal_resampling='causal source-token hold at20Hz' if os.environ['XR_REPLAY_VARIANT']=='physics200' else 'none')
            token_file.write_text(json.dumps(token_report,indent=2))
            return wrapped
        evaluation.train_agent_trl.create_manager_env = create
    runtime_adapter.install = installation


def make_haptics(env):
    """Execute upstream unmodified force signal and driver against 2.3 sensor views."""
    import numpy as np
    import torch
    spec = importlib.util.spec_from_file_location('upstream_haptic_feedback', UPSTREAM)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    class HandView:
        def __init__(self, prefix):
            self.sensors = [v for k,v in env.scene.sensors.items() if k.startswith(prefix) and k.removeprefix(prefix).isdigit()]
            self.body_names = [s.body_names[0] for s in self.sensors]
        @property
        def data(self):
            forces = []
            for sensor in self.sensors:
                channel = next(i for i,p in enumerate(sensor.cfg.filter_prim_paths_expr) if p.endswith('/Object'))
                forces.append(sensor.data.force_matrix_w[:, :, channel:channel+1, :])
            return SimpleNamespace(normal_force_matrix_w=SimpleNamespace(torch=torch.cat(forces, dim=1)))

    class RecordingReceiver:
        def __init__(self):
            self.latest = {}
        def send_haptic(self, endpoint, values):
            self.latest[endpoint] = np.array(values, copy=True)
    proxy = SimpleNamespace(scene={'left_hand_contact':HandView('table_watch_left_'),
                                  'right_hand_contact':HandView('grasp_finger_')})
    receiver = RecordingReceiver()
    cfg = module.GloveHapticFeedbackCfg()
    return module.HapticFeedbackDriver(proxy, receiver, cfg), receiver
