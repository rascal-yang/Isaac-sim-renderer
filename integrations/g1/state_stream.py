"""Complete registered-asset state recording at 200Hz and render packets at50Hz."""
import json,os,time,socket,struct
from pathlib import Path
import numpy as np

FIELDS=['root_state_w','body_pos_w','body_quat_w','body_vel_w']
JOINT_FIELDS=['joint_pos','joint_vel','joint_pos_target','joint_vel_target','joint_effort_target']

def values(env):
    rows={}
    for name,asset in {**env.scene.articulations,**env.scene.rigid_objects}.items():
        d=asset.data
        for key in FIELDS+(JOINT_FIELDS if name in env.scene.articulations else []):
            rows[name+'__'+key]=getattr(d,key).detach().cpu().numpy().copy()
    for name,view in getattr(env,"_pipeline_extra_views",{}).items():
        rows[name+"__transform_xyzw"]=view.get_transforms().detach().cpu().numpy().copy()
        rows[name+"__velocity"]=view.get_velocities().detach().cpu().numpy().copy()
    return rows

class StateRecorder:
    def __init__(self,capture):
        import omni.usd
        from pxr import UsdPhysics,UsdGeom
        self.capture=capture;self.env=capture.env;self.out=Path(capture.out);self.count=0;self.socket=None
        env=self.env
        assert not env.scene.deformable_objects and not env.scene.rigid_object_collections
        stage=omni.usd.get_context().get_stage();prefixes=[]
        for name,asset in {**env.scene.articulations,**env.scene.rigid_objects}.items():
            prefixes.append(asset.cfg.prim_path.replace('{ENV_REGEX_NS}','/World/envs/env_0').replace('env_.*','env_0'))
        rigid=[];uncovered=[];static=[];disabled=[]
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI) and UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get():
                path=str(prim.GetPath());rigid.append(path)
                if not any(path==q or path.startswith(q+'/') for q in prefixes):uncovered.append(path)
            if prim.IsA(UsdGeom.Xform) and not any(str(prim.GetPath())==q or str(prim.GetPath()).startswith(q+'/') for q in prefixes):
                xf=UsdGeom.Xformable(prim)
                static.append(dict(path=str(prim.GetPath()),matrix=np.asarray(xf.GetLocalTransformation()).tolist(),reset_stack=bool(xf.GetResetXformStack())))
        disabled=[str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI) and not UsdPhysics.RigidBodyAPI(p).GetRigidBodyEnabledAttr().Get()]
        env._pipeline_extra_views={}
        physics_view=next(iter(env.scene.articulations.values()))._physics_sim_view
        extras={}
        for i,path in enumerate(uncovered):
            name='extra_'+str(i);view=physics_view.create_rigid_body_view(path)
            assert view.count==1,(path,view.count)
            env._pipeline_extra_views[name]=view;extras[name]=path
        static=[x for x in static if not any(x['path']==q or x['path'].startswith(q+'/') for q in uncovered)]
        sample=values(env);self.dtype=np.dtype([(k,v.dtype,v.shape) for k,v in sample.items()])
        self.physics=np.lib.format.open_memmap(self.out/'full_physics_states.npy',mode='w+',dtype=self.dtype,shape=(int(os.environ['G1_MAX_STEPS'])*4+1,))
        self.frames=np.lib.format.open_memmap(self.out/'frame_states.npy',mode='w+',dtype=self.dtype,shape=(int(os.environ['G1_MAX_STEPS'])+1,))
        self.contacts=np.lib.format.open_memmap(self.out/'contact_force_matrix_w.npy',mode='w+',dtype=np.float32,shape=(int(os.environ['G1_MAX_STEPS'])*4+1,32,3,3))
        self.net_contacts=np.lib.format.open_memmap(self.out/'net_contact_forces_w.npy',mode='w+',dtype=np.float32,shape=(int(os.environ['G1_MAX_STEPS'])*4+1,32,3))
        self.cameras=np.lib.format.open_memmap(self.out/'camera_matrices.npy',mode='w+',dtype=np.float64,shape=(int(os.environ['G1_MAX_STEPS'])+1,4,4))
        self.schema=dict(articulations=list(env.scene.articulations),rigid_objects=list(env.scene.rigid_objects),
             rigid_body_paths=rigid,disabled_rigid_api_paths=disabled,extra_rigid_bodies=extras,uncovered_rigid_bodies=[],static_xforms=static,
             contact_records=dict(hz=200,rows=int(os.environ['G1_MAX_STEPS'])*4+1,world_frame=True,force_unit="N",filters="batch_contact_config.json"),
             fields=list(sample),physics_hz=200,frame_hz=50,physics_rows=int(os.environ['G1_MAX_STEPS'])*4+1,frame_rows=int(os.environ['G1_MAX_STEPS'])+1,
             state_scope='All registered roots, COM velocities, joints, targets, link poses and link velocities; 200Hz plus50Hz render packets. Static USD transforms and assets retained. PhysX private warm-start caches not exported; intended for rendering, not resuming dynamics.',
             shapes={k:list(v.shape) for k,v in sample.items()})
        (self.out/'state_schema.json').write_text(json.dumps(self.schema,indent=2))
        self.started=None;self.write_s=0.;self.publish_wait_s=0.;self.frame_times=[];self.queue_events=[];self.last_ack=-1
        old=env.scene.update
        def update(*a,**kw):
            result=old(*a,**kw)
            if capture.active:
                self.count+=1;start=time.perf_counter();self.assign(self.physics,self.count,values(env));self.write_s+=time.perf_counter()-start
            return result
        env.scene.update=update
    @staticmethod
    def assign(dest,index,row):
        for k,v in row.items():dest[k][index]=v
    def record_contacts(self):
        start=time.perf_counter();watch=self.capture.contact_watch
        self.contacts[self.count]=watch.matrix.detach().cpu().numpy()
        self.net_contacts[self.count]=watch.net.detach().cpu().numpy()
        self.write_s+=time.perf_counter()-start
    def initial(self):
        self.record_contacts()
        row=values(self.env);self.assign(self.physics,0,row);self.assign(self.frames,0,row)
        self.cameras[0]=np.asarray(self.capture.camera.Get())
        (self.out/'producer_ready.json').write_text(json.dumps(dict(pid=os.getpid(),ready_wall=time.time())))
        if os.environ.get('PIPE_STREAM')=='1':
            sock_path=os.environ['PIPE_SOCKET'];deadline=time.monotonic()+120
            while True:
                try:
                    s=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);s.connect(sock_path);self.socket=s;break
                except (FileNotFoundError,ConnectionRefusedError):
                    s.close();assert time.monotonic()<deadline;time.sleep(.05)
            assert self.socket.recv(1)==b'R'
        self.started=time.perf_counter();self.started_wall=time.time()
    def post(self):
        i=self.capture.control_index
        start=time.perf_counter();self.assign(self.frames,i+1,values(self.env))
        self.cameras[i+1]=np.asarray(self.capture.camera.Get());self.write_s+=time.perf_counter()-start
        self.frame_times.append([i,time.perf_counter(),time.time()])
        if self.socket is not None:
            start=time.perf_counter();self.socket.sendall(struct.pack('!q',i))
            if i-self.last_ack>=8:
                data=self.recv_exact(8);self.last_ack=struct.unpack('!q',data)[0]
            end=time.perf_counter();self.publish_wait_s+=end-start;self.queue_events.append([i,start,end])
    def recv_exact(self,n):
        out=b''
        while len(out)<n:
            chunk=self.socket.recv(n-len(out));assert chunk,'Render worker disconnected';out+=chunk
        return out
    def close(self):
        if self.socket is not None:
            start=time.perf_counter();self.socket.sendall(struct.pack('!q',-1))
            while self.last_ack<self.capture.control_index-1:
                self.last_ack=struct.unpack('!q',self.recv_exact(8))[0]
            self.socket.shutdown(socket.SHUT_WR)
            self.socket.close();self.publish_wait_s+=time.perf_counter()-start
        start=time.perf_counter()
        for z in [self.physics,self.frames,self.cameras,self.contacts,self.net_contacts]:z.flush()
        (self.out/'state_recording_timing.json').write_text(json.dumps(dict(physical_steps=self.count,state_capture_s=self.write_s,publish_wait_s=self.publish_wait_s,flush_s=time.perf_counter()-start,producer_start_perf=self.started,producer_start_wall=self.started_wall,producer_close_perf=time.perf_counter(),producer_close_wall=time.time(),queue_events=self.queue_events,frame_publications=self.frame_times),indent=2))
