"""Observe the existing synchronous PhysX boundary without changing its ordering."""
import json,os,time
from pathlib import Path
import numpy as np

def install(capture):
    import carb,omni.physx,torch
    import omni.physx.bindings._physx as pxb
    s=carb.settings.get_settings();out=Path(capture.out);ctx=capture.env.sim._physics_context
    iface=ctx._physx_sim_interface
    traced=os.environ.get('PHYSX_NSYS','0')=='1'
    bench=omni.physx.get_physx_benchmarks_interface()
    bench.enable_profile(True)
    keys=[pxb.SETTING_NUM_THREADS,pxb.SETTING_UPDATE_TO_USD,pxb.SETTING_UPDATE_VELOCITIES_TO_USD,
          pxb.SETTING_USE_ACTIVE_CUDA_CONTEXT,pxb.SETTING_EXPOSE_PROFILER_DATA,
          '/app/renderer/waitIdle','/app/hydraEngine/waitIdle','/app/asyncRendering']
    scene_api=ctx._physx_scene_api
    (out/'physx_boundary_config.json').write_text(json.dumps(dict(settings={k:s.get(k) for k in keys},
        nsys=traced,benchmarks_enabled=True,trace_control_range=[650,700],
        device=capture.env.device,gpu_dynamics=scene_api.GetEnableGPUDynamicsAttr().Get(),
        broadphase=scene_api.GetBroadphaseTypeAttr().Get(),solver=scene_api.GetSolverTypeAttr().Get()),indent=2))
    rows=[];stats_start=None;trace_running=False
    class Proxy:
        def __getattr__(self,name):return getattr(iface,name)
        def simulate(self,*a,**kw):
            nonlocal stats_start,trace_running
            if not capture.active:return iface.simulate(*a,**kw)
            if stats_start is None:stats_start=bench.get_profile_stats()
            if traced and not trace_running and capture.control_index==650:
                torch.cuda.cudart().cudaProfilerStart();trace_running=True
            active_trace=traced and trace_running and capture.control_index<700
            if active_trace:torch.cuda.nvtx.range_push('PHYSX_simulate')
            start=time.perf_counter()
            try:return iface.simulate(*a,**kw)
            finally:
                elapsed=time.perf_counter()-start
                if active_trace:torch.cuda.nvtx.range_pop()
                rows.append([capture.control_index,capture.env._sim_step_counter,elapsed,0.])
        def fetch_results(self,*a,**kw):
            nonlocal trace_running
            if not capture.active:return iface.fetch_results(*a,**kw)
            active_trace=traced and trace_running and capture.control_index<700
            if active_trace:torch.cuda.nvtx.range_push('PHYSX_fetch_results')
            start=time.perf_counter()
            try:return iface.fetch_results(*a,**kw)
            finally:
                rows[-1][3]=time.perf_counter()-start
                if active_trace:torch.cuda.nvtx.range_pop()
                if traced and trace_running and capture.control_index==699 and capture.env._sim_step_counter%4==0:
                    torch.cuda.cudart().cudaProfilerStop();trace_running=False
    ctx._physx_sim_interface=Proxy()
    original=capture.close
    def close(*a,**kw):
        if capture.closed:return
        stats=bench.get_profile_stats();bench.enable_profile(False)
        z=np.asarray(rows);np.save(out/'physx_step_times.npy',z)
        report=dict(steps=len(z),simulate_s=float(z[:,2].sum()),fetch_results_s=float(z[:,3].sum()),
                    benchmark_start=stats_start,benchmark_end=stats,
                    columns=['control_index','physics_step_index','simulate_s','fetch_results_s'])
        (out/'physx_boundary_profile.json').write_text(json.dumps(report,indent=2,default=str))
        return original(*a,**kw)
    capture.close=close
