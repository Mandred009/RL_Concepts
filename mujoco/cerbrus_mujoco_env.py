import gymnasium as gym
import numpy as np
import mujoco as mj
from mujoco.glfw import glfw
from gymnasium import spaces
import os
import time
import math

class MujocoGymEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, xml_path, simend=10.0, render_mode=None):
        super().__init__()

        # Load model
        full_path = os.path.join(os.path.dirname(__file__), xml_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"XML model not found: {full_path}")

        self.model = mj.MjModel.from_xml_path(full_path)
        self.data = mj.MjData(self.model)
        self.simend = simend
        self.render_mode = render_mode
        self.time_step = 1.0 / 60.0
        self.elapsed_steps = 0
        self.sensor_data=[0.0]*20 
        self.rot_euler=[0.0,0.0,0.0]
        self.min_z_ht=0.15 # if the main body of the quad falls below this height we terminate

        # Action and observation spaces
        self.action_space = spaces.Box(
            low=self.model.actuator_ctrlrange[:, 0],
            high=self.model.actuator_ctrlrange[:, 1],
            dtype=np.float32
        )

        self.obs_dim = self.model.nq + 20
        self.observation_space = spaces.Box(  # change this
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        # Initialize camera and other components if render_mode is "human"
        if self.render_mode == "human":
            glfw.init()
            self.window = glfw.create_window(1200, 900, "MuJoCo Gym Env", None, None)
            glfw.make_context_current(self.window)
            glfw.swap_interval(1)

            self.cam = mj.MjvCamera()
            mj.mjv_defaultCamera(self.cam)

            self.opt = mj.MjvOption()
            mj.mjv_defaultOption(self.opt)

            self.scene = mj.MjvScene(self.model, maxgeom=10000)
            self.context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)

            # Set up initial camera view (adjust to fit your model)
            self.cam.azimuth = 90
            self.cam.elevation = -45
            self.cam.distance = 2.5
            self.cam.lookat = np.array([0.0, 0.0, 1.0])

            # Set initial mouse button states
            self.button_left = False
            self.button_middle = False
            self.button_right = False
            self.lastx = 0
            self.lasty = 0

            # Set GLFW mouse and keyboard callbacks
            glfw.set_mouse_button_callback(self.window, self.mouse_button)
            glfw.set_cursor_pos_callback(self.window, self.mouse_move)
            glfw.set_scroll_callback(self.window, self.scroll)

    def _get_obs(self):
        self.sensor_data = self.data.sensordata

        rot_quat=self.sensor_data[16:]
        self.rot_euler=self.euler_from_quaternion(rot_quat[1],rot_quat[2],rot_quat[3],rot_quat[0])
        return np.concatenate([self.data.qpos, self.sensor_data[0:16]]).astype(np.float32) # all joint positions, acc_raw, gyro_raw, mag_raw

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mj.mj_resetData(self.model, self.data)
        self.elapsed_steps = 0
        self.sensor_data=[0.0]*20
        self.rot_euler=[0.0,0.0,0.0]
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:] = action

        mj.mj_step(self.model, self.data)
        self.elapsed_steps += 1

        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self.data.time >= self.simend
        
        if self.sensor_data[15]<=self.min_z_ht:
            truncated=True
        else:
            truncated = False
        info = {}

        return obs, reward, terminated, truncated, info

    def _compute_reward(self): # Engineer your rewards
        
        
        return 1.0

    def render(self):
        if self.render_mode != "human":
            return

        # Update camera and render the scene
        mj.mjv_updateScene(
            self.model, self.data, self.opt, None, self.cam,
            mj.mjtCatBit.mjCAT_ALL.value, self.scene
        )

        width, height = glfw.get_framebuffer_size(self.window)
        viewport = mj.MjrRect(0, 0, width, height)
        mj.mjr_render(viewport, self.scene, self.context)
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def close(self):
        if self.render_mode == "human":
            glfw.destroy_window(self.window)
            glfw.terminate()

    def mouse_button(self, window, button, act, mods):
        self.button_left = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        self.button_middle = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
        self.button_right = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        glfw.get_cursor_pos(window)

    def mouse_move(self, window, xpos, ypos):
        dx = xpos - self.lastx
        dy = ypos - self.lasty
        self.lastx, self.lasty = xpos, ypos

        if not (self.button_left or self.button_middle or self.button_right):
            return

        width, height = glfw.get_window_size(window)
        mod_shift = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or \
                    glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS

        if self.button_right:
            action = mj.mjtMouse.mjMOUSE_MOVE_H if mod_shift else mj.mjtMouse.mjMOUSE_MOVE_V
        elif self.button_left:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H if mod_shift else mj.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = mj.mjtMouse.mjMOUSE_ZOOM

        mj.mjv_moveCamera(self.model, action, dx / height, dy / height, self.scene, self.cam)

    def scroll(self, window, xoffset, yoffset):
        mj.mjv_moveCamera(self.model, mj.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.05 * yoffset, self.scene, self.cam)
    
    def euler_from_quaternion(self,x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = round(math.atan2(t0, t1),2)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = round(math.asin(t2),2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = round(math.atan2(t3, t4),2)
     
        return roll_x, pitch_y, yaw_z # in radians


###--- Example Usage ---###

if __name__ == "__main__":
    env = MujocoGymEnv(xml_path="cerbrus_scene.xml", render_mode="human")
    obs, _ = env.reset()
    action = [0]*12
    for i in range(500):
        print(i)
        action=[x+0.01 for x in action]
        obs, reward, done, truncated, info = env.step(action)
        if truncated:
            break
        env.render()
    env.close()
