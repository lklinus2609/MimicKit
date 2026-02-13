"""
Video Recorder for Wandb Logging (Headless)

Records policy rollouts as videos and logs them to Weights & Biases.
Uses Newton's headless rendering - no display required.
"""

import os
os.environ.setdefault("PYGLET_HEADLESS", "1")

import numpy as np
import torch

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from newton.viewer import ViewerGL
    NEWTON_AVAILABLE = True
except ImportError:
    NEWTON_AVAILABLE = False

import util.mp_util as mp_util


class HeadlessVideoRecorder:
    """
    Headless video recorder for MimicKit training.

    Creates a separate headless viewer that attaches to the training
    environment's existing model and state. Only records on root process.
    """

    def __init__(self, env, agent, device, width=640, height=480, fps=30):
        self._env = env
        self._agent = agent
        self._device = device
        self._width = width
        self._height = height
        self._fps = fps

        self._viewer = None
        self._initialized = False

    def _ensure_viewer(self):
        """Create headless viewer and attach to training model."""
        if self._initialized:
            return self._viewer is not None

        self._initialized = True

        if not NEWTON_AVAILABLE:
            print("[VideoRecorder] Newton viewer not available")
            return False

        engine = getattr(self._env, '_engine', None)
        if engine is None:
            print("[VideoRecorder] No _engine on environment")
            return False

        sim_model = getattr(engine, '_sim_model', None)
        if sim_model is None:
            print("[VideoRecorder] No _sim_model on engine")
            return False

        try:
            self._viewer = ViewerGL(
                width=self._width,
                height=self._height,
                headless=True
            )
            self._viewer.set_model(sim_model)

            env_spacing = getattr(engine, '_env_spacing', None)
            if env_spacing is not None:
                self._viewer.set_world_offsets([env_spacing, env_spacing, 0.0])

            print(f"[VideoRecorder] Created headless viewer {self._width}x{self._height}")
            return True

        except Exception as e:
            print(f"[VideoRecorder] Failed to create viewer: {e}")
            self._viewer = None
            return False

    def _get_raw_state(self):
        """Get the current Newton raw_state from the environment's engine."""
        engine = getattr(self._env, '_engine', None)
        if engine is None:
            return None
        sim_state = getattr(engine, '_sim_state', None)
        if sim_state is None:
            return None
        return getattr(sim_state, 'raw_state', None)

    def _capture_frame(self):
        """Capture a single frame from the current simulation state."""
        if self._viewer is None:
            return None

        state = self._get_raw_state()
        if state is None:
            return None

        try:
            self._viewer.begin_frame(0.0)
            self._viewer.log_state(state)
            self._viewer.end_frame()

            frame_wp = self._viewer.get_frame()
            if frame_wp is None:
                return None

            return frame_wp.numpy().copy()
        except Exception as e:
            if not hasattr(self, '_capture_error_printed'):
                print(f"[VideoRecorder] Frame capture error: {e}")
                self._capture_error_printed = True
            return None

    def record_and_log(self, iteration, sample_count=None, max_steps=300):
        """
        Record one episode using the agent's policy and log to wandb.

        Temporarily switches agent to TEST mode for deterministic actions,
        then restores original mode. Only runs on root process.

        Args:
            iteration: Current training iteration (for print messages)
            sample_count: Wandb step value (must match the step used by wandb_logger).
                          If None, logs without an explicit step.
            max_steps: Maximum steps per recorded episode.
        """
        if not mp_util.is_root_proc():
            return

        if not WANDB_AVAILABLE or wandb.run is None:
            return

        if not self._ensure_viewer():
            return

        frames = []

        # Import here to avoid circular imports
        import learning.base_agent as base_agent
        import envs.base_env as base_env

        # Save and switch to test mode for deterministic actions
        prev_mode = self._agent._mode
        self._agent.set_mode(base_agent.AgentMode.TEST)

        try:
            obs, info = self._env.reset()

            for step in range(max_steps):
                with torch.no_grad():
                    action, _ = self._agent._decide_action(obs, info)

                obs, reward, done, info = self._env.step(action)

                frame = self._capture_frame()
                if frame is not None:
                    frames.append(frame)

                # Check if first env is done
                if torch.is_tensor(done):
                    if done[0].item() != base_env.DoneFlags.NULL.value:
                        break
                elif done:
                    break

        finally:
            # Restore previous mode
            self._agent.set_mode(prev_mode)

        if not frames:
            print(f"[VideoRecorder] No frames captured at iteration {iteration}")
            return

        video = np.stack(frames, axis=0)  # (T, H, W, C)

        log_dict = {"policy_video": wandb.Video(video, fps=self._fps, format="mp4")}
        if sample_count is not None:
            wandb.log(log_dict, step=int(sample_count))
        else:
            wandb.log(log_dict)

        print(f"[VideoRecorder] Logged video at iter {iteration} ({len(frames)} frames, {video.shape})")
