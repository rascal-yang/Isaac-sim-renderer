#!/usr/bin/env python3
"""Portable entry points for the validated G1 recorded-state renderer."""
import argparse, hashlib, importlib.util, json, os, shutil, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def state_sizes(n):
    if not isinstance(n,int) or n<1: raise ValueError('token count must be positive')
    return n*4+1,n+1

def split_cores(groups):
    if len(groups)<2: raise ValueError('至少需要两个可用物理核心')
    # Physical cores stay intact, including SMT siblings. Match the measured
    # machine by distributing both large SMT groups and single-thread cores.
    a=[]; b=[]
    for size in sorted({len(g) for g in groups},reverse=True):
        cohort=[g for g in groups if len(g)==size];cut=(len(cohort)+1)//2
        a.extend(x for g in cohort[:cut] for x in g)
        b.extend(x for g in cohort[cut:] for x in g)
    if not b: a=[x for g in groups[:len(groups)//2] for x in g];b=[x for g in groups[len(groups)//2:] for x in g]
    return sorted(a),sorted(b)

def cpu_groups():
    groups={}
    for c in sorted(os.sched_getaffinity(0)):
        p=Path(f'/sys/devices/system/cpu/cpu{c}/topology')
        key=((p/'physical_package_id').read_text().strip(),(p/'core_id').read_text().strip())
        groups.setdefault(key,[]).append(c)
    return list(groups.values())

def resource_package(path):
    package=Path(path).resolve()/'sonic-grasp-augmentation'
    if not (package/'resources.json').is_file():
        raise FileNotFoundError('缺少交付资源：请解压完整交付包，或用 --resources 指向含 sonic-grasp-augmentation 的目录。Git 仓库不含模型与资产。')
    return package

def dependency(resources):
    path=resource_package(resources)/'launch.py'
    spec=importlib.util.spec_from_file_location('g1_resource_delivery',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def prepare(resources,runtime):
    manifest=json.loads((Path(resources)/'manifest.json').read_text())
    assert digest(resource_package(resources)/'resources.json')==manifest['g1_resources_manifest_sha256'],'G1 resource manifest differs'
    for name,expected in manifest['extra_files'].items():
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts: raise ValueError('Unsafe resource path')
        assert digest(Path(resources)/relative)==expected,'Resource checksum mismatch: '+name
    dep=dependency(resources);dep.prepare(runtime)
    dst=runtime/'workspace/experiments/state_render_pipeline'
    shutil.copytree(ROOT/'integrations/g1',dst,dirs_exist_ok=True)
    return dst

def compare_physical(source,reference):
    names=['full_physics_states.npy','frame_states.npy','camera_matrices.npy','contact_force_matrix_w.npy','net_contact_forces_w.npy','haptic_feedback.npz','measured_states.npz','decoder_io.npz','collision_forces.npz']
    return {name:digest(source/name)==digest(reference/name) for name in names}

def summarize(output):
    out=Path(output);process=json.loads((out/'process_report.json').read_text())
    source=out/'baseline' if process['scheme']=='baseline' else out/'physics'
    render=json.loads((out/'render/render_profile.json').read_text()) if (out/'render/render_profile.json').exists() else None
    result={'scheme':process['scheme'],'cold_process_s':process['total_process_wall_s'],'allocation':process['allocation']}
    if not source.exists():
        result.update(active_s=sum(render[k] for k in ['warmup_s','render_loop_wall_s','finalize_s']),scope='render only; source physics excluded')
    else:
        prof=json.loads((source/'profile.json').read_text())
        source_total=sum(prof[k] for k in ['rollout_wall_s','existing_finalize_s','probe_artifact_write_s'])
        active=source_total
        if render:
            active=max(prof['artifacts_finished_perf'],render['render_finish_perf'])-prof['rollout_start_perf'] if process['scheme']=='overlap' else source_total+sum(render[k] for k in ['warmup_s','render_loop_wall_s','finalize_s'])
        result.update(active_s=active,source_execution_and_finalize_s=source_total,physics_s=prof['methods']['physics_sim_step']['inclusive_s'],sonic_s=prof['methods']['sonic_decoder_onnx']['inclusive_s'],task_success=json.loads((source/'lean_result.json').read_text())['task_success'],scope='first control through source artifacts/render finalization; overlapping stages not summed')
    if render:result['renderer']=dict(warmup_s=render['warmup_s'],loop_s=render['render_loop_wall_s'],finalize_s=render['finalize_s'],timing=render['timing'])
    (out/'timing_summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2));return result

def verify(output,reference=None):
    import numpy as np
    out=Path(output);source=out/'physics' if (out/'physics').exists() else out/'baseline' if (out/'baseline').exists() else Path(json.loads((out/'process_report.json').read_text())['source'])
    rendered=out/'render' if (out/'render').exists() else out/'baseline'
    schema=json.loads((source/'state_schema.json').read_text());n=schema['frame_rows']-1
    assert schema['physics_rows']==n*4+1
    for name,rows in [('frame_states.npy',n+1),('camera_matrices.npy',n+1),('full_physics_states.npy',n*4+1),('contact_force_matrix_w.npy',n*4+1),('net_contact_forces_w.npy',n*4+1)]:
        a=np.load(source/name,mmap_mode='r',allow_pickle=False);assert len(a)==rows,(name,a.shape,rows)
    video=rendered/'head.mp4'
    data=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_streams','-of','json',str(video)]))['streams'][0]
    assert int(data['nb_read_frames'])==n and [data['width'],data['height']]==[640,480]
    assert data['avg_frame_rate']=='50/1'
    subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(video),'-f','null','-'],check=True,stdout=subprocess.DEVNULL)
    report={'frames':n,'duration_s':n/50,'video_full_decode_passed':True,'resolution':[640,480],'physical_steps':n*4}
    if (rendered/'render_profile.json').exists():
        p=json.loads((rendered/'render_profile.json').read_text());assert p['physics_steps_during_render']==0
        assert p['source_rows']==list(range(1,n+1))
        assert max(x[2] for x in p['completed_camera_records'])<1e-8
        assert p['static_xform_max_abs_error']<1e-8
        assert max(p['state_readback_max_abs_errors'].values())<1e-4
        report.update(physics_steps_during_render=0,camera_max_abs_error=max(x[2] for x in p['completed_camera_records']),state_max_abs_error=max(p['state_readback_max_abs_errors'].values()))
    if reference:
        reference=Path(reference); reference=reference/'physics' if (reference/'physics').exists() else reference/'baseline' if (reference/'baseline').exists() else reference
        report['reference_byte_identical']=compare_physical(source,reference)
        assert all(report['reference_byte_identical'].values()), report
    (out/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));return report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--resources',type=Path,default=ROOT/'resources');s=p.add_subparsers(dest='command',required=True)
    s.add_parser('doctor')
    q=s.add_parser('prepare');q.add_argument('--runtime',type=Path,required=True)
    for name in ['pipeline','render']:
        q=s.add_parser(name);q.add_argument('--runtime',type=Path,required=True);q.add_argument('--output',type=Path,required=True);q.add_argument('--candidate',type=Path);q.add_argument('--timeout',type=int,default=1800)
        if name=='pipeline':q.add_argument('--scheme',choices=['baseline','sequential','overlap'],default='overlap')
        else:q.add_argument('--source',type=Path,required=True)
    q=s.add_parser('verify');q.add_argument('--output',type=Path,required=True);q.add_argument('--reference',type=Path)
    a=p.parse_args()
    if a.command=='verify':verify(a.output,a.reference);return
    dep=dependency(a.resources)
    if a.command=='doctor':
        import importlib.metadata as metadata
        versions={}
        for name in ['isaacsim','isaaclab','numpy','torch','onnxruntime','psutil']:
            try: versions[name]=metadata.version(name)
            except metadata.PackageNotFoundError:
                if name!='onnxruntime': raise
                probe_env=os.environ.copy();probe_env['PYTHONPATH']=str(a.resources.resolve()/'onnxruntime')
                versions[name]=subprocess.check_output([sys.executable,'-c','import onnxruntime; print(onnxruntime.__version__)'],env=probe_env,text=True).strip()
        assert sys.version_info[:2]==(3,11), '需要 Python 3.11'
        assert versions['isaacsim']=='5.1.0.0', versions
        assert shutil.which('ffmpeg') and shutil.which('ffprobe'), '请安装系统 ffmpeg'
        subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv'],check=True)
        print(json.dumps({'installed_versions':versions},indent=2))
        groups=cpu_groups();left,right=split_cores(groups)
        print(json.dumps({'resource_package':str(resource_package(a.resources)),'available_cpus':len(os.sched_getaffinity(0)),'physics_cpus':left,'render_cpus':right,'kit_processes_require_gpu':True,'clean_machine_install_tested':False},indent=2));return
    runtime=a.runtime.resolve()
    if a.command=='prepare':prepare(a.resources,runtime);return
    if not (runtime/'repository_runtime.json').is_file():raise RuntimeError('先运行 prepare --runtime 指向同一目录')
    # Validate the prepared resources before replay. Paths are materialized once.
    worker=prepare(a.resources,runtime)
    candidate=(a.candidate or a.resources/'demo_candidate').resolve()
    import numpy as np
    n=len(np.load(candidate/'grasp_tokens.npz')['time_s']);state_sizes(n)
    env=dep.environment(runtime,ort_path=a.resources.resolve()/'onnxruntime');env['RENDER_G1_RUNTIME']=str(runtime)
    left,right=split_cores(cpu_groups())
    env.update(RENDER_PHYSICS_CPUS=','.join(map(str,left)),RENDER_GPU_CPUS=','.join(map(str,right)),RENDER_PYTHON=sys.executable,RENDER_CANDIDATE=str(candidate),RENDER_TIMEOUT=str(a.timeout))
    output=a.output.resolve()
    if 'robot' in str(output):raise ValueError('上游路径限制：输出路径不能含 robot 子串')
    if output.exists():raise FileExistsError(f'输出目录必须是新目录: {output}')
    cmd=[sys.executable,str(ROOT/'coordinator.py'),str(output),'--scheme','sequential' if a.command=='render' else a.scheme]
    if a.command=='render':cmd.extend(['--source',str(a.source.resolve())])
    subprocess.run(cmd,env=env,cwd=runtime/'workspace',check=True)
    verify(output)
    summarize(output)
if __name__=='__main__':main()
