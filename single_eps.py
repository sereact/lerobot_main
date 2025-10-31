import h5py
import shutil
import numpy as np
# --- Input/output files ---
src_path = "/home/ubuntu/mount-point/libero_regenerate_retry/libero1/libero_object_reg/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5"
path = "/home/ubuntu/mount-point/libero_regenerate_retry/libero1/libero_object_reg/demo_001.hdf5"

demo_to_copy = "demo_7"
demo_key = "data/demo_7/actions"
def print_hdf5_structure(name, obj):
    print(name)
# with h5py.File(path, "r") as f:
#     if demo_key not in f:
#         print(f"❌ Dataset {demo_key} not found. Available top keys:", list(f.keys()))
#     else:
#         actions = f[demo_key][()]  # load dataset into numpy array
#         print(f"✅ Loaded {demo_key}: shape = {actions.shape}, dtype = {actions.dtype}")

#         # Show statistics
#         print("Min:", np.min(actions))
#         print("Max:", np.max(actions))
#         print("Mean:", np.mean(actions))
#         print("First few actions:\n", actions[:5])
with h5py.File(path, "r") as f:
    print("=== HDF5 File Structure ===")
    f.visititems(print_hdf5_structure)

# with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
#     if "data" not in src:
#         raise ValueError("Expected 'data' group not found in source file.")

#     # List available demos (groups)
#     demos = list(src["data"].keys())
#     print(f"Found {len(demos)} demos: {demos[:10]}")

#     if demo_to_copy not in demos:
#         raise ValueError(f"Demo {demo_to_copy} not found in /data. Available: {demos}")

#     print(f"Extracting {demo_to_copy} ...")

#     # Create "data" group in destination to preserve structure
#     dst.create_group("data")

#     # Copy the selected demo
#     src["data"].copy(demo_to_copy, dst["data"])


# print(f"✅ Saved {demo_to_copy} to {dst_path}")
