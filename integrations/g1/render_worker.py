"""Restore recorded scene states and render without advancing dynamics."""
import json,os,time,socket,struct
from pathlib import Path
import numpy as np

def run(capture):
    import torch,omni.usd
    import omni.replicator.core as rep
    from pxr import Gf,UsdGeom,UsdPhysics
    env=capture.env;source=Path(os.environ['PIPE_SOURCE']);out=Path(capture.out)
    deadline=time.monotonic()+120
    while not (source/'state_schema.json').exists():
        assert time.monotonic()<deadline;time.sleep(.05)
    schema=json.loads((source/'state_schema.json').read_text())
    assert schema['articulations']==list(env.scene.articulations)
    assert schema['rigid_objects']==list(env.scene.rigid_objects)
    frames=np.load(source/'frame_states.npy',mmap_mode='r');cameras=np.load(source/'camera_matrices.npy',mmap_mode='r')
    stage=omni.usd.get_context().get_stage();static_error=0.
    physics_view=next(iter(env.scene.articulations.values()))._physics_sim_view
    extras={name:physics_view.create_rigid_body_view(path) for name,path in schema['extra_rigid_bodies'].items()}
    indices=torch.tensor([0],dtype=torch.int32)

    for item in schema['static_xforms']:
        if item['path'].startswith(('/World/Capture','/OmniverseKit_Persp')):continue
        prim=stage.GetPrimAtPath(item['path']);assert prim.IsValid(),item['path']
        static_error=max(static_error,float(np.max(abs(np.asarray(UsdGeom.Xformable(prim).GetLocalTransformation())-np.asarray(item['matrix'])))))
    assert static_error<1e-8,static_error
    assets={**env.scene.articulations,**env.scene.rigid_objects}
    kinematic={name:bool(UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(asset.cfg.prim_path.replace('{ENV_REGEX_NS}','/World/envs/env_0').replace('env_.*','env_0'))).GetKinematicEnabledAttr().Get()) for name,asset in env.scene.rigid_objects.items()}

    timings={'restore_s':0.,'forward_s':0.,'render_s':0.,'rgb_s':0.,'wait_s':0.,'raw_rgb_s':0.}
    errors={k:0. for k in frames.dtype.names};events=[];delivered=[];socket_server=None;conn=None
    stream=os.environ.get('PIPE_STREAM')=='1'
    if stream:
        socket_server=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);socket_server.bind(os.environ['PIPE_SOCKET']);socket_server.listen(1)
        (out/'renderer_ready.json').write_text(json.dumps(dict(pid=os.getpid(),ready_wall=time.time())))
        conn,_=socket_server.accept();conn.sendall(b'R')
    else:
        while not (source/'lean_result.json').exists():
            assert time.monotonic()<deadline;time.sleep(.05)
    def restore(row,mat):
        start=time.perf_counter()
        for name,asset in assets.items():
            root=torch.from_numpy(row[name+'__root_state_w'].copy())
            if kinematic.get(name,False):asset.write_root_pose_to_sim(root[:,:7])
            else:asset.write_root_state_to_sim(root)
            if name in env.scene.articulations:
                asset.write_joint_state_to_sim(torch.from_numpy(row[name+'__joint_pos'].copy()),torch.from_numpy(row[name+'__joint_vel'].copy()))
                for field,method in [('joint_pos_target','set_joint_position_target'),('joint_vel_target','set_joint_velocity_target'),('joint_effort_target','set_joint_effort_target')]:
                    getattr(asset,method)(torch.from_numpy(row[name+'__'+field].copy()))
        for name,view in extras.items():
            view.set_transforms(torch.from_numpy(row[name+'__transform_xyzw'].copy()),indices)
            view.set_velocities(torch.from_numpy(row[name+'__velocity'].copy()),indices)
        capture.camera.Set(Gf.Matrix4d(*np.asarray(mat).ravel().tolist()))
        timings['restore_s']+=time.perf_counter()-start
    def read_check(row):
        # FK readback checks all links as well as stored roots, joints and targets.
        for name,asset in assets.items():
            for field in [k.split('__',1)[1] for k in errors if k.startswith(name+'__')]:
                actual=getattr(asset.data,field).detach().cpu().numpy()
                key=name+'__'+field;error=float(np.max(abs(actual-row[key])));errors[key]=max(errors[key],error)
        for name,view in extras.items():
            for field,method in [('transform_xyzw','get_transforms'),('velocity','get_velocities')]:
                key=name+'__'+field;errors[key]=max(errors[key],float(np.max(abs(getattr(view,method)().numpy()-row[key]))))
    # Render-only phase must never submit another physics simulate call.
    ctx=env.sim._physics_context;old_step=ctx._step
    def forbidden(*a,**kw):raise RuntimeError('Physics advancement forbidden in recorded-state renderer')
    ctx._step=forbidden
    while not (source/'producer_ready.json').exists():
        assert time.monotonic()<deadline;time.sleep(.05)
    cam_anno=rep.annotators.get("camera_params");cam_anno.attach(capture.render_product)
    start=time.perf_counter();restore(frames[0],cameras[0])
    for _ in range(4):env.sim.render()
    capture.audit_renderer('renderer_settings.json');read_check(frames[0]);warmup=time.perf_counter()-start
    raw=np.lib.format.open_memmap(out/'raw_rgb.npy',mode='w+',dtype=np.uint8,shape=(schema['frame_rows']-1,480,640,3))
    # Metadata follows the completed render; camera matrix is independently checked.
    metadata=[];detail_events=[]
    start=time.perf_counter();started_wall=time.time()
    for expected in range(schema['frame_rows']-1):
        step_start=time.perf_counter();wait_start=step_start
        if conn:
            data=b''
            while len(data)<8:
                block=conn.recv(8-len(data));assert block,'Producer disconnected';data+=block
            index=struct.unpack('!q',data)[0]
        else:index=expected
        wait_end=time.perf_counter();timings['wait_s']+=wait_end-wait_start
        assert index==expected,(index,expected)
        row=frames[index+1];mat=cameras[index+1]
        restore(row,mat);restore_end=time.perf_counter()
        # sim.render includes forward/Fabric exactly once, as in the original path.
        t=time.perf_counter();env.sim.render();timings['render_s']+=time.perf_counter()-t;render_end=time.perf_counter()
        t=time.perf_counter();capture.control_index=index;capture.read_rgb();timings['rgb_s']+=time.perf_counter()-t;rgb_end=time.perf_counter()
        t=time.perf_counter();raw[index]=capture.rgb;timings['raw_rgb_s']+=time.perf_counter()-t
        # Every-frame state and completed camera verification, separate CPU audit cost.
        read_check(row)
        camera_view=np.asarray(cam_anno.get_data()['cameraViewTransform']).reshape(4,4)
        camera_error=float(np.max(abs(np.linalg.inv(camera_view)-mat)))
        metadata.append([index,(index+1)*.02,camera_error])
        detail_events.append([index,step_start,wait_end,restore_end,render_end,rgb_end,time.perf_counter()])
        delivered.append(index);events.append([index,step_start,time.perf_counter(),time.time()])
        if conn:conn.sendall(struct.pack('!q',index))
        if (index+1)%250==0:print('RENDER_PROGRESS',index+1,flush=True)
    if conn:
        data=b""
        while len(data)<8:
            block=conn.recv(8-len(data));assert block;data+=block
        assert struct.unpack("!q",data)[0]==-1
    loop_s=time.perf_counter()-start
    t=time.perf_counter();capture.writer.close();raw.flush();finalize_s=time.perf_counter()-t
    capture.audit_renderer('renderer_settings_end.json')
    np.savez_compressed(out/'image_timestamps.npz',control_index_and_time=np.asarray(capture.image_rows))
    if conn:
        # All frame IDs arrived; end marker can remain queued during producer close.
        conn.close();socket_server.close()
    assert max(x[2] for x in metadata)<1e-8
    report=dict(warmup_s=warmup,render_loop_wall_s=loop_s,finalize_s=finalize_s,render_start_wall=started_wall,render_start_perf=start,render_finish_wall=time.time(),render_finish_perf=time.perf_counter(),
        timing=timings,frames=len(delivered),physics_steps_during_render=0,state_readback_max_abs_errors=errors,
        detail_events=detail_events,static_xform_max_abs_error=static_error,completed_camera_records=metadata,events=events,
        source=str(source),source_rows=[i+1 for i in delivered],scope='Recorded-state rendering only; task/contact/haptic results come from the source physical execution.')
    (out/'render_profile.json').write_text(json.dumps(report,indent=2))
    print('RENDER_DONE',loop_s,finalize_s,'state max',max(errors.values()),flush=True)
