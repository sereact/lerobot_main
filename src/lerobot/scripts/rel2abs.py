from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from PIL import Image
import os
import numpy as np
import imageio
import robosuite.utils.transform_utils as T
from tqdm import tqdm
import h5py
import types
import numpy as np
from scipy.spatial.transform import Rotation as R


def get_camera_intrinsics(env, cam_name: str, width: int, height: int) -> np.ndarray:
    """Compute intrinsic matrix from camera parameters (MuJoCo FOV)."""
    cam_id = env.sim.model.camera_name2id(cam_name)
    fovy = float(env.sim.model.cam_fovy[cam_id])  # degrees
    fovy_rad = np.deg2rad(fovy)

    f_y = (height / 2.0) / np.tan(fovy_rad / 2.0)
    f_x = f_y  # assume square pixels
    c_x = width / 2.0
    c_y = height / 2.0

    K = np.array(
        [[f_x, 0, c_x],
        [0, f_y, c_y],
        [0, 0, 1]],
        dtype=np.float32
    )
    return K

def eval_libero_action(task_suite, task_id, demo_id, video_save_path, libero_raw_data_dir):
    task = task_suite.get_task(task_id)
    task_name = task.name
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    # print(f"[info] retrieving task {task_id} from suite {task_suite_name}, the " + \
    #     f"language instruction is {task_description}, and the bddl file is {task_bddl_file}")
    
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": 480,
        "camera_widths": 480,
        "camera_depths": True
    }
    
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    print(env)
    orig_data_path = os.path.join(libero_raw_data_dir, f"{task.name}_demo.hdf5")
    orig_data_file = h5py.File(orig_data_path, "r")
    orig_data = orig_data_file["data"]
    demo_data = orig_data[f"demo_{demo_id}"]
    orig_states = demo_data["states"][()]
    obs = env.reset()
    env.set_init_state(orig_states[0])
    actions = demo_data["actions"]
    
    for _ in range(10):
        obs, reward, done, info = env.step(np.array([0, 0, 0, 0, 0, 0, -1]))

    images = [np.flip(np.flip(obs['agentview_image'], 0), 1)]
    depths_agent = []
    depths_robot = []

    abs_eef = []
    intrinsics_agentview = get_camera_intrinsics(env, "agentview", 480, 480)
    intrinsics_robot = get_camera_intrinsics(env, "robot0_eye_in_hand", 480, 480)

    for action in actions:
        obs, reward, done, info = env.step(action)
        goal_pos = env.env.robots[0].controller.goal_pos
        goal_ori = env.env.robots[0].controller.goal_ori

        state_ori = T.quat2axisangle(T.mat2quat(env.env.robots[0].controller.goal_ori))
        abs_eef.append(np.concatenate([goal_pos, state_ori, action[-1:]]))
        images.append(np.flip(np.flip(obs['agentview_image'], 0), 1))
        depths_agent.append(obs["agentview_depth"].copy())
        depths_robot.append(obs["robot0_eye_in_hand_depth"].copy())


    env.close()
    print("done")
    os.makedirs(video_save_path, exist_ok=True)
    if demo_id <= 10: 
        imageio.mimsave(os.path.join(video_save_path, f"demo_{demo_id}.mp4"), images, fps=30)
    print(task_name, task_description, reward, done)
    orig_data_file.close()
    with h5py.File(orig_data_path, "a") as f:
        grp = f["data"][f"demo_{demo_id}"]
        grp_a = grp["obs"]
        if "absolute_actions" in grp:
            del grp["absolute_actions"]
        grp.create_dataset("absolute_actions", data=abs_eef)
        grp_a.create_dataset("agentview_depth", data=depths_agent)
        grp_a.create_dataset("robot0_eye_in_hand_depth", data=depths_robot)
        grp.create_dataset("intrinsics_agentview", data=intrinsics_agentview)
        grp.create_dataset("intrinsics_robot0_eye_in_hand", data=intrinsics_robot)

        return np.stack(abs_eef), done


def eval_libero_abs_action(task_suite, task_id, demo_id, video_save_path, states, libero_raw_data_dir):
    task = task_suite.get_task(task_id)
    task_name = task.name
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    # print(f"[info] retrieving task {task_id} from suite {task_suite_name}, the " + \
    #     f"language instruction is {task_description}, and the bddl file is {task_bddl_file}")

    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": 256,
        "camera_widths": 256
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    orig_data_path = os.path.join(libero_raw_data_dir, f"{task.name}_demo.hdf5")
    orig_data_file = h5py.File(orig_data_path, "r")
    orig_data = orig_data_file["data"]
    demo_data = orig_data[f"demo_{demo_id}"]
    orig_states = demo_data["states"][()]
    env.reset()
    env.set_init_state(orig_states[0])
    for _ in range(10):
        obs, reward, done, info = env.step(np.array([0, 0, 0, 0, 0, 0, -1]))
    for robot in env.env.robots:
        robot.controller.use_delta=False

    images = [np.flip(np.flip(obs['agentview_image'], 0), 1)]
    for state in states:
        print(state)
        obs, reward, done, info = env.step(state)
        images.append(np.flip(np.flip(obs['agentview_image'], 0), 1))

    env.close()

    os.makedirs(video_save_path, exist_ok=True)
    imageio.mimsave(os.path.join(video_save_path, f"demo_{demo_id}.mp4"), images, fps=30)
    print(task_name, task_description, reward, done)

benchmark_dict = benchmark.get_benchmark_dict()
data_root = '/home/ubuntu/git/sereact/dataset_libero_raw/libero_object'

for task_suite_name in tqdm(os.listdir(data_root)):
    task_suite = benchmark_dict["libero_object"]()
    print(task_suite.tasks)
    for task_id in tqdm(range(len(task_suite.tasks))):
        
        print("task_id", task_id)
        task = task_suite.get_task(task_id)
        task_name = task.name
        for i in tqdm(range(50)):
            # action_path = os.path.join(data_root, task_suite_name, f"{task_name}_demo", f"demo_{i}", "action.npy")
            # action = np.load(action_path)
            libero_raw_data_dir = '/home/ubuntu/git/sereact/dataset_libero_raw/libero_object/libero_object'

            video_save_path = '/home/ubuntu/git/sereact/lerobot/src/lerobot/scripts/test/rel'
            abs_eef, done = eval_libero_action(task_suite, task_id, i, video_save_path, libero_raw_data_dir)

            # video_save_path = '/home/ubuntu/git/sereact/lerobot/src/lerobot/scripts/test/abs'
            # eval_libero_abs_action(task_suite, task_id, i, video_save_path, abs_eef, libero_raw_data_dir)
        # exit()