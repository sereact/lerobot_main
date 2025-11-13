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

from multiprocessing import Process
import os

def eval_libero_action(task_suite, task_id, demo_id, video_save_path, libero_raw_data_dir):
    task = task_suite.get_task(task_id)
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
    print(env)
    orig_data_path = os.path.join(libero_raw_data_dir, f"{task.name}_demo.hdf5")
    orig_data_file = h5py.File(orig_data_path, "r")
    orig_data = orig_data_file["data"]
    demo_data = orig_data[f"demo_{demo_id}"]
    orig_states = demo_data["states"][()].copy()
    obs = env.reset()
    env.set_init_state(orig_states[0])
    actions = demo_data["actions"][()].copy()
    
    for _ in range(10):
        obs, reward, done, info = env.step(np.array([0, 0, 0, 0, 0, 0, -1]))

    images = [np.flip(np.flip(obs['agentview_image'], 0), 1)]
    abs_eef = []

    for action in actions:
        obs, reward, done, info = env.step(action)
        goal_pos = env.env.robots[0].controller.goal_pos
        goal_ori = env.env.robots[0].controller.goal_ori

        state_ori = T.quat2axisangle(T.mat2quat(env.env.robots[0].controller.goal_ori))
        abs_eef.append(np.concatenate([goal_pos, state_ori, action[-1:]]))
        images.append(np.flip(np.flip(obs['agentview_image'], 0), 1))
    env.close()
    print("done")
    os.makedirs(video_save_path, exist_ok=True)
    if demo_id <= 10: 
        imageio.mimsave(os.path.join(video_save_path, f"demo_{demo_id}.mp4"), images, fps=30)
    print(task_description, reward, done)
    orig_data_file.close()
    with h5py.File(orig_data_path, "a") as f:
        grp = f["data"][f"demo_{demo_id}"]
        if "absolute_actions" in grp:
            del grp["absolute_actions"]
        grp.create_dataset("absolute_actions", data=abs_eef)

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

def process_suite(root, gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    suite_name = os.path.basename(root)
    task_suite = benchmark_dict[suite_name]()
    print(f"Processing suite: {suite_name}")

    libero_raw_data_dir = os.path.join(root, suite_name)

    for task_id in range(len(task_suite.tasks)):
        print(f"[{suite_name}] Task {task_id}")
        for demo_id in range(50):
            abs_eef, done = eval_libero_action(
                task_suite,
                task_id,
                demo_id,
                "/home/ubuntu/git/sereact/lerobot/src/lerobot/scripts/test/rel",
                libero_raw_data_dir
            )

if __name__ == "__main__":
    data_roots = ['/home/ubuntu/git/sereact/dataset_libero_raw/libero_spatial', '/home/ubuntu/git/sereact/dataset_libero_raw/libero_object', '/home/ubuntu/git/sereact/dataset_libero_raw/libero_10', '/home/ubuntu/git/sereact/dataset_libero_raw/libero_goal']

    benchmark_dict = benchmark.get_benchmark_dict()

    processes = []
    for gpu_id, root in enumerate(data_roots):
        p = Process(target=process_suite, args=(root, gpu_id))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

# for root in data_roots:
#     suite_name = os.path.basename(root) 
#     task_suite = benchmark_dict[suite_name]()
#     print(task_suite.tasks)
#     for task_id in tqdm(range(len(task_suite.tasks))):

#         print("task_id", task_id)
#         task = task_suite.get_task(task_id)
#         task_name = task.name
#         for i in tqdm(range(50)):
#             # action_path = os.path.join(data_root, task_suite_name, f"{task_name}_demo", f"demo_{i}", "action.npy")
#             # action = np.load(action_path)
#             libero_raw_data_dir = '/home/ubuntu/git/sereact/dataset_libero_raw/libero_spatial/libero_spatial'

#             video_save_path = '/home/ubuntu/git/sereact/lerobot/src/lerobot/scripts/test/rel'
#             abs_eef, done = eval_libero_action(task_suite, task_id, i, video_save_path, libero_raw_data_dir)

#             # video_save_path = '/home/ubuntu/git/sereact/lerobot/src/lerobot/scripts/test/abs'
#             # eval_libero_abs_action(task_suite, task_id, i, video_save_path, abs_eef, libero_raw_data_dir)
#         # exit()