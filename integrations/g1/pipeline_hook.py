"""Separate source dynamics from render-only state restoration."""
import os,time
from pathlib import Path

def install():
    import runtime_adapter
    original_install=runtime_adapter.install
    def installation(evaluation,out,intern):
        original_install(evaluation,out,intern)
        original_create=evaluation.train_agent_trl.create_manager_env
        def create(*a,**kw):
            wrapped=original_create(*a,**kw)
            if os.environ.get('PIPE_ROLE')=='physics' and os.environ.get('PIPE_STREAM')=='1':
                # Treat the second Kit setup as startup, before the first control step.
                ready=Path(out).parent/'render'/'renderer_ready.json';deadline=time.monotonic()+120
                while not ready.exists():
                    assert time.monotonic()<deadline,'Render worker initialization timeout'
                    time.sleep(.05)
            if os.environ.get('PIPE_ROLE')=='render':
                from render_worker import run
                run(wrapped.env._xr_capture)
                os._exit(0)  # Match the existing isolated Isaac evaluation exit path.
            return wrapped
        evaluation.train_agent_trl.create_manager_env=create
    runtime_adapter.install=installation
