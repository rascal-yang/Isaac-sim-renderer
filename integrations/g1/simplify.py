"""Fixed-token-only removal of diagnostics and inactive policy plumbing."""
import json
import os
from pathlib import Path


def install_simplifications(wrapped, out):
    import torch
    env = wrapped.env
    capture = env._xr_capture
    assert os.environ['G1_LEAN_MODE'] == 'replay'
    assert os.environ['G1_GRAIL_ACTIVE'] == '0'
    assert not os.environ.get('G1_COMPARE_MODE')
    assert not env.cfg.commands.motion.debug_vis
    assert not env.command_manager.get_term("motion").use_adaptive_sampling
    assert env.num_envs == 1 and env.physics_dt == .005 and env.step_dt == .02
    assert env._grasp_extension.controller == 'sonic_full'
    # Terminations depend on reference poses/time, but none of the enabled terms
    # reads these ten diagnostic metrics. Keep original command.compute/update.
    forbidden = ('cumm_body_errors', 'cumm_body_pos_error', 'cumm_body_ori_error',
                 'cumm_body_pos_error_local')
    assert not any(n in env.termination_manager.active_terms for n in forbidden)
    command = env.command_manager.get_term('motion')
    old_metrics = command._update_metrics
    def update_metrics():
        if not capture.active:
            return old_metrics()
    command._update_metrics = update_metrics

    # This collector deliberately leaves legacy RL observations frozen. The actual
    # release decoder reads current robot.data directly, never this dictionary.
    old_prepare = wrapped._prepare_obs_for_action_transform_module
    def prepare(obs):
        return {} if capture.active else old_prepare(obs)
    wrapped._prepare_obs_for_action_transform_module = prepare
    old_process = wrapped.process_raw_obs
    def process(raw_obs, *args, **kwargs):
        if capture.active and raw_obs is env.obs_buf:
            return wrapped._last_obs_dict
        return old_process(raw_obs, *args, **kwargs)
    wrapped.process_raw_obs = process

    # replay_controller already replaced this legacy body policy with zeros.
    # Reuse its zero body placeholder; the real 29 targets are still supplied by
    # the release decoder in term.apply_actions at every physical substep.
    zero_body = torch.zeros((1, 29), device=env.device)
    wrapped.action_transform_module.forward = lambda *a, **kw: zero_body
    env._simplified_fixed_replay = True
    (Path(out)/'simplifications.json').write_text(json.dumps(dict(
        enabled=True,
        skipped=['unused reference tracking metrics during rollout',
                 'inactive legacy ATM observation preparation',
                 'reprocessing frozen legacy RL observations',
                 'disabled GRAIL residual copying/assertion and no-op planner tick'],
        retained=['reference clock and pose updates', 'all termination/supervisor checks',
                  'actual SONIC live proprioception and inference', 'every-step action/target writes',
                  'full measured state/decoder/contact/haptic records', 'same RGB capture and video'],
        invalid_diagnostics=['motion.metrics during rollout; not refreshed'],
        scope='Only fixed-token replay with GRAIL disabled; not an RL training environment'),indent=2))
