"""Remove one-env CUDA housekeeping and unused implicit torque diagnostics."""
import json
import os
from pathlib import Path


def install_runtime(env, out):
    import torch
    from isaaclab.actuators import ImplicitActuator
    from isaaclab.sensors import ContactSensor
    capture = env._xr_capture
    audit = os.environ.get('RUNTIME_COMMAND_AUDIT', '0') == '1'
    report = dict(sensor_refresh=[], actuator_models={}, audit_enabled=audit,
                  approximate_rl_torque_diagnostics_produced=False,
                  actual_physx_PD_and_limits_unchanged=True)
    env._fast_command_audit = dict(comparisons=0, max_target_error=0.)

    for name, sensor in env.scene.sensors.items():
        if name.startswith(('grasp_finger_', 'table_watch_')):
            continue  # Already maintained by the all-substep batch contact view.
        if sensor._num_envs != 1 or sensor.cfg.update_period != 0 or sensor.cfg.debug_vis:
            continue
        original_reset = sensor.reset
        valid = {'value':False}

        def refresh(s=sensor, v=valid):
            # Same original numerical buffer update, including contact histories,
            # airtime and frame transforms. Known all-env CPU ids avoid CUDA nonzero.
            if not v['value']:
                s._update_buffers_impl([0])
                s._timestamp_last_update.copy_(s._timestamp)
                s._is_outdated.zero_()
                v['value'] = True

        def update(dt, force_recompute=False, s=sensor, v=valid, fn=refresh):
            s._timestamp += dt
            v['value'] = False
            fn()

        def reset(env_ids=None, original=original_reset, v=valid):
            original(env_ids)
            v['value'] = False

        sensor.update = update
        sensor.reset = reset
        sensor._update_outdated_buffers = refresh
        report['sensor_refresh'].append(dict(name=name, type=type(sensor).__name__,
              physics_hz=200, history_length=sensor.cfg.history_length,
              track_air_time=getattr(sensor.cfg,'track_air_time',False)))

    for name, asset in env.scene.articulations.items():
        classes = {k:type(v).__name__ for k,v in asset.actuators.items()}
        report['actuator_models'][name] = classes
        if not all(type(actuator) is ImplicitActuator for actuator in asset.actuators.values()):
            continue
        ids = []
        for actuator in asset.actuators.values():
            indices = actuator.joint_indices
            ids.extend(list(range(asset.num_joints))[indices] if isinstance(indices,slice) else list(indices))
            asset._data.soft_joint_vel_limits[:,indices] = actuator.velocity_limit
        assert sorted(ids) == list(range(asset.num_joints)), (name, ids)
        original_apply = asset._apply_actuator_model

        def apply(a=asset, original=original_apply):
            # ImplicitActuator.compute returns these targets unchanged. Physical
            # drive feedback and saturation continue inside PhysX every 5 ms.
            a._joint_pos_target_sim.copy_(a._data.joint_pos_target)
            a._joint_vel_target_sim.copy_(a._data.joint_vel_target)
            a._joint_effort_target_sim.copy_(a._data.joint_effort_target)
            if audit and capture.active and env._sim_step_counter % 4 == 1:
                expected = [t.clone() for t in [a._joint_pos_target_sim,a._joint_vel_target_sim,a._joint_effort_target_sim]]
                original()
                actual = [a._joint_pos_target_sim,a._joint_vel_target_sim,a._joint_effort_target_sim]
                error = float(torch.stack([(x-y).abs().max() for x,y in zip(expected,actual)]).max())
                env._fast_command_audit['comparisons'] += 1
                env._fast_command_audit['max_target_error'] = max(env._fast_command_audit['max_target_error'],error)
                assert error == 0., ('Implicit actuator target mismatch', error)
        asset._apply_actuator_model = apply
    (Path(out)/'streamline_config.json').write_text(json.dumps(report,indent=2))

    previous_close = capture.close
    def close():
        previous_close()
        (Path(out)/'command_target_audit.json').write_text(json.dumps(env._fast_command_audit,indent=2))
    capture.close = close
