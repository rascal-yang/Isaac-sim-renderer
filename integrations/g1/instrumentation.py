"""Experiment-only wall-clock profiling. No explicit CUDA synchronization."""
import collections
import functools
import json
import os
from pathlib import Path
import time
import numpy as np

class Meter:
    def __init__(self):
        self.enabled = False
        self.stack = []
        self.samples = collections.defaultdict(list)
    def call(self, name, fn, *a, **kw):
        if not self.enabled:
            return fn(*a, **kw)
        frame = [time.perf_counter(), 0.]
        self.stack.append(frame)
        try:
            return fn(*a, **kw)
        finally:
            elapsed = time.perf_counter()-frame[0]
            self.stack.pop()
            if self.stack: self.stack[-1][1] += elapsed
            self.samples[name].append((elapsed, elapsed-frame[1]))
    def wrap(self, obj, attr, name):
        old = getattr(obj, attr)
        @functools.wraps(old)
        def wrapped(*a, **kw): return self.call(name, old, *a, **kw)
        setattr(obj, attr, wrapped)
    def report(self):
        return {k: dict(calls=len(v), inclusive_s=sum(x[0] for x in v), exclusive_s=sum(x[1] for x in v), mean_ms=1000*np.mean([x[0] for x in v]), p95_ms=1000*np.percentile([x[0] for x in v],95), max_ms=1000*max(x[0] for x in v)) for k,v in self.samples.items()}

def install():
    import runtime_adapter
    old_install = runtime_adapter.install
    def installation(evaluation, out, intern):
        old_install(evaluation, out, intern)
        old_create = evaluation.train_agent_trl.create_manager_env
        def create(*a, **kw):
            wrapped = old_create(*a, **kw)
            attach(wrapped, Path(out))
            return wrapped
        evaluation.train_agent_trl.create_manager_env = create
    runtime_adapter.install = installation

def attach(wrapped, out):
    import torch
    import onnxruntime as ort
    from training_capture import TrainingCapture
    from contact_watch import ContactWatch
    from automatic_policy import AutoGraspPolicy
    from isaaclab.sensors import ContactSensor
    env = wrapped.env
    ext = env._grasp_extension
    meter = Meter()
    detailed = os.environ.get('PROFILE_DETAIL','1') == '1'
    camera = os.environ.get('PROFILE_CAMERA','0') == '1'
    state = dict(steps=0, first=None, last=None, rows=[], inputs=[], outputs=[], control_events=[], initial_saved=False)
    def sample():
        row={}
        for name in ['robot','brainco_left','brainco_right','object']:
            d=env.scene[name].data
            for attr in ['root_state_w','joint_pos','joint_vel','body_pos_w','body_quat_w','joint_pos_target']:
                if hasattr(d,attr): row[name+'__'+attr]=getattr(d,attr)[0].detach().cpu().numpy().copy()
        return row
    def checked_sample(): return meter.call('probe_pose_record',sample)
    # Records only actual ONNX inputs/outputs, never substitutes state or actions.
    original_run=ort.InferenceSession.run
    def run(session,*a,**kw):
        if not meter.enabled: return original_run(session,*a,**kw)
        result=meter.call('sonic_decoder_onnx', original_run,session,*a,**kw) if detailed else original_run(session,*a,**kw)
        feed=a[1] if len(a)>1 else kw['input_feed']
        def save():
            state['inputs'].append(next(iter(feed.values()))[0].copy())
            state['outputs'].append(result[0][0].copy())
        meter.call('probe_decoder_record',save)
        return result
    ort.InferenceSession.run=run
    old_begin=TrainingCapture.begin
    def begin(c):
        if not state['initial_saved']:
            state['rows'].append(checked_sample());state['initial_saved']=True
        old_begin(c)
    TrainingCapture.begin=begin
    old_post=TrainingCapture.post_step
    def post(c):
        old_post(c)
        state['rows'].append(checked_sample())
    TrainingCapture.post_step=post
    if detailed:
        for obj,attr,label in [
            (ext,'before_step','controller_before_step'),(ext,'metrics','grasp_metrics'),
            (env.sim,'step','physics_sim_step'),(env.sim,'render','sim_render'),
            (env.scene,'update','scene_update'),(env.scene,'write_data_to_sim','scene_write_commands'),
            (env,'step','isaac_env_step'),
            (TrainingCapture,'begin','capture_begin'),(TrainingCapture,'post_step','capture_post_step'),
            (TrainingCapture,'end','capture_end'),
            (ContactWatch,'sample','contact_watch_sample'),(ContactWatch,'forces','contact_forces'),
            (ContactWatch,'refresh_contacts','batch_contact_refresh'),
            (ContactSensor,'_update_buffers_impl','contact_sensor_refresh'),
            (AutoGraspPolicy,'decide','numeric_supervisor'),
            (torch.Tensor,'cpu','tensor_to_cpu')]:
            meter.wrap(obj,attr,label)
        for manager,attrs in [('action_manager',['process_action','apply_action']),('observation_manager',['compute']),('termination_manager',['compute']),('reward_manager',['compute']),('command_manager',['compute']),('recorder_manager',['record_post_step','record_post_physics_decimation_step'])]:
            for attr in attrs: meter.wrap(getattr(env,manager),attr,manager+'.'+attr)
    if detailed:
        meter.wrap(env.sim._physics_context, '_step', 'physx_context_step')
        meter.wrap(ext.env._xr_capture.haptic_driver, 'update', 'official_haptic_driver')
        meter.wrap(ext.env._xr_capture, 'read_rgb', 'rgb_read_encode')
    original_step=wrapped.step
    def step(*a,**kw):
        if state['first'] is None: state['first']=time.perf_counter()
        control_start=time.perf_counter()
        meter.enabled=True
        try: return meter.call('wrapped_control_step',original_step,*a,**kw)
        finally:
            meter.enabled=False;state['control_events'].append([state['steps'],control_start,time.perf_counter()]);state['steps']+=1;state['last']=time.perf_counter()
    wrapped.step=step
    old_end=wrapped.end_render_results
    def end(*a,**kw):
        if (out/'profile.json').exists(): return old_end(*a,**kw)
        start=time.perf_counter()
        result=old_end(*a,**kw)
        close_s=time.perf_counter()-start
        save_start=time.perf_counter()
        if state['rows']:
            np.savez_compressed(out/'measured_states.npz', **{k:np.stack([r[k] for r in state['rows']]) for k in state['rows'][0]})
        np.savez_compressed(out/'decoder_io.npz',inputs=np.asarray(state['inputs']),outputs=np.asarray(state['outputs']))
        methods=meter.report()
        rollout=state['last']-state['first'] if state['first'] else None
        report=dict(control_steps=state['steps'],samples=len(state['rows']),camera=camera,detailed=detailed,rollout_wall_s=rollout,
            control_step_wall_s=methods.get('wrapped_control_step',{}).get('inclusive_s'),
            between_control_steps_s=rollout-methods.get('wrapped_control_step',{}).get('inclusive_s',0) if rollout else None,
            control_events=state["control_events"],rollout_start_perf=state["first"],rollout_end_perf=state["last"],artifacts_finished_perf=time.perf_counter(),
            existing_finalize_s=close_s,probe_artifact_write_s=time.perf_counter()-save_start,methods=methods,
            physics_dt=env.physics_dt, control_dt=env.step_dt, variant=os.environ.get('XR_REPLAY_VARIANT'),
            semantics='RT2/native DLAA 640x480, one synchronous render update per post-physics state at50Hz, four startup warmups. Physics200Hz/SONIC50Hz unchanged. Original contact buffer/history numerics retained with one-env CPU dirty flags. PhysX implicit PD unchanged; unused approximate RL torque diagnostics omitted. Nested exclusive/inclusive CPU wall clock, no added CUDA sync in timing runs. GPU waits may be charged at later blocking calls. Initial + all pre-reset actual states. Fixed tokens, no GPT/encoder/planner calls. Single-update image latency not certified.')
        (out/'profile.json').write_text(json.dumps(report,indent=2))
        (out/'pose_schema.json').write_text(json.dumps({name:dict(joint_names=getattr(env.scene[name],'joint_names',[]),body_names=getattr(env.scene[name],'body_names',[])) for name in ['robot','brainco_left','brainco_right','object']},indent=2))
        return result
    wrapped.end_render_results=end
