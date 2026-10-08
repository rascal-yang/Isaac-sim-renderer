"""Same collision guards on merged, single-body filtered Isaac Lab sensors."""
import json
import os
from pathlib import Path
import numpy as np
import torch


class ContactWatch:
    def __init__(self, env, capture, out):
        self.env, self.capture, self.out = env, capture, Path(out)
        self.threshold, self.latched, self.next_alert_step = .5, None, 0
        self.substep_count, self.positive_samples, self.peak = 0, 0, 0.
        self.sensors, self.labels, self.rows = [], [], []
        for name,sensor in env.scene.sensors.items():
            if not name.startswith(('grasp_finger_', 'table_watch_')):
                continue
            channels=[]
            for i,p in enumerate(sensor.cfg.filter_prim_paths_expr):
                if p.endswith(('/Table','/Room')):
                    channels.append(i)
                    self.labels.append(p.rsplit('/',1)[-1]+':'+sensor.body_names[0])
            self.sensors.append((sensor,channels))
        self.audit_queries = os.environ.get('BATCH_CONTACT_AUDIT', '0') == '1'
        self.audit_max_error, self.audit_count = 0., 0
        self.alias_bound = False
        self.install_batch_view()
        old_update = env.scene.update
        def update(*a, **kw):
            result = old_update(*a, **kw)
            self.refresh_contacts()
            if capture.active:
                self.sample()
            return result
        env.scene.update = update
        capture.contact_watch = self
        (self.out/'contact_watch_config.json').write_text(json.dumps(dict(frequency_hz=1/env.physics_dt,
            threshold_total_table_force_N=.5, sensor_count=len(self.sensors), monitored_channels=len(self.labels),
            links=self.labels, response='Latch on every physics update; stop at next control checkpoint'),indent=2))

    def install_batch_view(self):
        # PhysX supports exact body paths + one filter list per path. Isaac Lab's
        # one-body filtered ContactSensor wrapper is not a PhysX API restriction.
        patterns, filters, guard_indices = [], [], []
        for index, (sensor, channels) in enumerate(self.sensors):
            assert sensor.num_bodies == 1 and not sensor.cfg.track_air_time
            patterns.append(sensor.body_physx_view.prim_paths[0])
            paths = [p.replace('.*', '*') for p in sensor.cfg.filter_prim_paths_expr]
            if len(paths) == 2:
                paths.append('/World/envs/env_*/Object')
            assert len(paths) == 3
            filters.append(paths)
            guard_indices.extend(index*3+channel for channel in channels)
        source = self.sensors[0][0]._physics_sim_view
        self.batch_view = source.create_rigid_contact_view(patterns, filter_patterns=filters)
        assert self.batch_view.sensor_count == len(patterns)
        assert self.batch_view.filter_count == 3
        self.guard_indices = torch.tensor(guard_indices, dtype=torch.long, device=self.env.device)
        self.refresh_contacts()
        # These aliases are refreshed once per physical step by the batch view.
        # Leave inherited six contact sensors on the ordinary Isaac Lab path.
        for sensor, _ in self.sensors:
            sensor.update = lambda dt, force_recompute=False: None
            sensor._update_outdated_buffers = lambda: None
        (self.out/'batch_contact_config.json').write_text(json.dumps(dict(
            sensor_paths=patterns, filters=filters, matrix_shape=[len(patterns),3,3],
            guard_indices=guard_indices, guard_hz=200, haptic_hz=50,
            audit_enabled=self.audit_queries), indent=2))

    def refresh_contacts(self):
        matrix = self.batch_view.get_contact_force_matrix(dt=self.env.physics_dt)
        net = self.batch_view.get_net_contact_forces(dt=self.env.physics_dt)
        if not self.alias_bound:
            self.matrix, self.net = matrix, net
            for i, (sensor, _) in enumerate(self.sensors):
                count = len(sensor.cfg.filter_prim_paths_expr)
                sensor._data.force_matrix_w = matrix[i:i+1, :count].unsqueeze(0)
                sensor._data.net_forces_w = net[i:i+1].unsqueeze(0)
                sensor._data.force_matrix_w_history = sensor._data.force_matrix_w.unsqueeze(1)
                sensor._data.net_forces_w_history = sensor._data.net_forces_w.unsqueeze(1)
            self.alias_bound = True
        else:
            assert matrix.data_ptr() == self.matrix.data_ptr() and net.data_ptr() == self.net.data_ptr()
        if self.audit_queries and self.capture.active and self.env._sim_step_counter % 4 == 0:
            errors = []
            for i, (sensor, _) in enumerate(self.sensors):
                original = sensor.contact_physx_view.get_contact_force_matrix(dt=self.env.physics_dt)
                errors.append((original-matrix[i, :original.shape[1]]).abs().max())
            error = float(torch.stack(errors).max())
            self.audit_max_error = max(self.audit_max_error, error)
            self.audit_count += 1
            assert error == 0., ('Batch force query changed values', error)

    def forces(self):
        vectors=self.matrix.reshape(-1,3).index_select(0,self.guard_indices)
        values=torch.linalg.vector_norm(vectors,dim=-1).detach().cpu().numpy()
        return dict(zip(self.labels, map(float,values)))

    def sample(self):
        if self.capture.state_recorder is not None:self.capture.state_recorder.record_contacts()
        values=self.forces(); total=sum(values.values())
        self.rows.append(list(values.values()))
        self.substep_count += 1
        self.peak=max(self.peak,total); self.positive_samples += total>.05
        if total>self.threshold and self.latched is None and self.capture.control_index>=self.next_alert_step:
            self.latched=dict(physics_index=self.substep_count-1, control_index=self.capture.control_index,
                time_s=self.substep_count*self.env.physics_dt,total_table_force_N=total,
                links_N={k:v for k,v in values.items() if v>.001})
            with (self.out/'contact_alerts.jsonl').open('a') as f:
                f.write(json.dumps(self.latched)+'\n')

    def observation(self):
        return dict(alert=self.latched,current_table_forces_N=self.forces(),
                    peak_total_table_force_N=self.peak,physics_samples=self.substep_count)

    def acknowledge(self, step):
        if self.latched is not None:
            self.next_alert_step=step+max(1,round(.2/self.env.step_dt))
        self.latched=None

    def close(self):
        (self.out/'batch_contact_audit.json').write_text(json.dumps(dict(
            compared_control_samples=self.audit_count, max_force_component_error_N=self.audit_max_error,
            enabled=self.audit_queries), indent=2))
        np.savez_compressed(self.out/'collision_forces.npz',forces_N=np.asarray(self.rows),
            links=np.array(self.labels),time_s=np.arange(1,len(self.rows)+1)*self.env.physics_dt)
