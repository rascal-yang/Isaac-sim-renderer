"""Fixed-token collector: physical operations retained, unused RL diagnostics omitted."""
import json
from pathlib import Path


def install():
    import runtime_adapter
    previous_install = runtime_adapter.install

    def installation(evaluation, out, intern):
        previous_install(evaluation, out, intern)
        previous_create = evaluation.train_agent_trl.create_manager_env

        def create(*args, **kwargs):
            wrapped = previous_create(*args, **kwargs)
            env = wrapped.env
            assert env.physics_dt == .005 and env.step_dt == .02 and env.cfg.decimation == 4
            import torch
            env.reward_buf = torch.zeros(env.num_envs, device=env.device)
            if __import__('os').environ.get('FAST_RUNTIME_GENERATION', 'fast') == 'fast':
                from streamline import install_runtime
                install_runtime(env, out)
                from simplify import install_simplifications
                install_simplifications(wrapped, out)

            def collector_step(action):
                # Same physical ordering as Isaac Lab 2.3.2 ManagerBasedRLEnv.step.
                env.action_manager.process_action(action.to(env.device))
                env.recorder_manager.record_pre_step()
                rendering = env.sim.has_gui() or env.sim.has_rtx_sensors()
                for _ in range(env.cfg.decimation):
                    env._sim_step_counter += 1
                    env.action_manager.apply_action()
                    env.scene.write_data_to_sim()
                    env.sim.step(render=False)
                    env.recorder_manager.record_post_physics_decimation_step()
                    if env._sim_step_counter % env.cfg.sim.render_interval == 0 and rendering:
                        env.sim.render()
                    env.scene.update(dt=env.physics_dt)
                env.episode_length_buf += 1
                env.common_step_counter += 1
                env.reset_buf = env.termination_manager.compute()
                env.reset_terminated = env.termination_manager.terminated
                env.reset_time_outs = env.termination_manager.time_outs
                env.reward_buf.zero_()
                # Capture reads actual physics state, contacts and RGB directly.
                if env.recorder_manager.active_terms:
                    env.recorder_manager.record_post_step()
                reset_ids = env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
                if len(reset_ids):
                    env.recorder_manager.record_pre_reset(reset_ids)
                    env._reset_idx(reset_ids)
                    if env.sim.has_rtx_sensors() and env.cfg.num_rerenders_on_reset > 0:
                        for _ in range(env.cfg.num_rerenders_on_reset):
                            env.sim.render()
                    env.recorder_manager.record_post_reset(reset_ids)
                env.command_manager.compute(dt=env.step_dt)
                if 'interval' in env.event_manager.available_modes:
                    env.event_manager.apply(mode='interval', dt=env.step_dt)
                # Legacy actor observations are placeholders after reset. They do not
                # enter the release decoder, which reads live proprioception at 50 Hz.
                return env.obs_buf, env.reward_buf, env.reset_terminated, env.reset_time_outs, env.extras

            env.step = collector_step
            (Path(out)/'collector_optimizations.json').write_text(json.dumps(dict(
                physics_hz=200, sonic_hz=50, collision_hz=200,
                omitted='Unused legacy RL reward and observation computations during rollout only',
                retained='Live SONIC proprioception, command schedule, all termination/supervisor checks, physical ordering, RGB and haptic records',
                intended_use='Fixed-token data replay; not an RL training environment'), indent=2))
            return wrapped
        evaluation.train_agent_trl.create_manager_env = create
    runtime_adapter.install = installation
