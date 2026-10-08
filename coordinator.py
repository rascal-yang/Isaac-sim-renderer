"""One trajectory, source CPU dynamics and a recorded-state GPU renderer."""
import argparse,fcntl,json,os,subprocess,time,signal,hashlib,psutil,shutil
from pathlib import Path
P=argparse.ArgumentParser();P.add_argument('name');P.add_argument('--scheme',choices=['baseline','sequential','overlap'],default='baseline');P.add_argument('--source');args=P.parse_args()
ROOT=Path(__file__).resolve().parent;W=Path(os.environ['RENDER_G1_RUNTIME'])/'workspace';PY=os.environ['RENDER_PYTHON'];WORKER=W/'experiments/state_render_pipeline'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
lock=(W/'experiments/.sonic_augmentation_gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
out=Path(args.name).resolve();assert not out.exists();out.mkdir(parents=True)
snapshot=out/'source_snapshot';snapshot.mkdir()
source_hashes={}
for p in WORKER.rglob('*.py'):
    shutil.copy2(p,snapshot/p.name);source_hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
sock=f'/tmp/g1pipe_{os.getpid()}.sock' ;children={};results={};start=time.perf_counter();start_wall=time.time();resources=[];last_gpu=0;gpu=None;track={}
def launch(role,source=None):
    target=out/role;env=os.environ.copy()
    env.update(G1_DRIVER_REEXEC='1',PROFILE_CAMERA='1' if role!='physics' else '0',PROFILE_DETAIL='1',BATCH_CONTACT_AUDIT='0',RUNTIME_COMMAND_AUDIT='0',PHYSX_TEST_MODE='cpu',PHYSX_NSYS='0',ACCELERATE_USE_CPU='true',PYTHONUNBUFFERED='1',PIPE_ROLE=role,PIPE_COORDINATED='1',PIPE_STREAM='1' if args.scheme=='overlap' else '0',PIPE_SOCKET=sock)
    if source:env['PIPE_SOURCE']=str(source)
    if args.scheme=='overlap':
        # Separate physical cores; both workers retain eight PhysX threads.
        cpus=os.environ['RENDER_PHYSICS_CPUS'] if role=='physics' else os.environ['RENDER_GPU_CPUS']
        env.update(PIPE_AFFINITY=cpus,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
        prefix=['taskset','-c',cpus]
    else:prefix=[]
    cmd=prefix+[PY,str(WORKER/'replay_xr.py'),'--candidate',os.environ['RENDER_CANDIDATE'],'--output',str(target),'--runtime','fast']
    log=(out/(role+'_launch.log')).open('w');process=subprocess.Popen(cmd,cwd=W,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    children[role]=(process,log,time.perf_counter());return process
try:
    if args.scheme=='baseline':launch('baseline')
    elif args.scheme=='sequential':
        if args.source:
            source=Path(args.source).resolve();launch('render',source)
        else:
            source=out/'physics';launch('physics')
    else:
        source=out/'physics';launch('render',source);launch('physics')
    pending=set(children)
    while pending:
        now=time.perf_counter()
        if now-last_gpu>2:
            q=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,utilization.memory','--format=csv,noheader,nounits'],capture_output=True,text=True)
            gpu=q.stdout.strip();last_gpu=now
        sample={'elapsed_s':now-start,'gpu':gpu,'processes':{}}
        for name,(proc,_,_) in children.items():
            try:
                pp=psutil.Process(proc.pid);tree=[pp]+pp.children(recursive=True);rss=cpu=0.
                for item in tree:
                    if item.pid not in track:track[item.pid]=item;item.cpu_percent()
                    cpu+=track[item.pid].cpu_percent();rss+=item.memory_info().rss
                sample['processes'][name]={'cpu_percent_sum':cpu,'rss_bytes_sum':rss}
            except psutil.Error:pass
        resources.append(sample)
        for role in list(pending):
            process,log,began=children[role]
            if process.poll() is not None:
                log.close();results[role]=dict(exit_code=process.returncode,process_wall_s=time.perf_counter()-began,pid=process.pid)
                assert process.returncode==0,(role,process.returncode)
                pending.remove(role)
                if args.scheme=='sequential' and role=='physics':launch('render',source);pending.add('render')
            else:
                runlog=out/role/'run.log'
                if runlog.exists() and 'Traceback (most recent call last)' in runlog.read_text()[-18000:]:raise RuntimeError(f'{role}: see {runlog}')
        assert time.perf_counter()-start<int(os.environ['RENDER_TIMEOUT']),'Pipeline timeout'
        time.sleep(.5)
finally:
    for role,(process,log,began) in children.items():
        if process.poll() is None:
            os.killpg(process.pid,signal.SIGKILL)
            process.wait()
        log.close()
    if Path(sock).exists():Path(sock).unlink()
report=dict(scheme=args.scheme,total_process_wall_s=time.perf_counter()-start,start_wall=start_wall,children=results,
    source=str(source) if args.scheme!='baseline' else None,
    allocation={'physics':os.environ['RENDER_PHYSICS_CPUS'],'render':os.environ['RENDER_GPU_CPUS']+' + GPU0'} if args.scheme=='overlap' else 'unrestricted original affinity',
    source_sha256=source_hashes)
(out/'resource_samples.json').write_text(json.dumps(resources,indent=2))
(out/'process_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
