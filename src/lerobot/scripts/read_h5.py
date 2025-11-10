import h5py
filename = "/home/ubuntu/mount-point/libero_abs_eef/libero_depth/libero_10_reg/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_demo.hdf5"
def print_hdf5_structure(name, obj):
    print(name)

with h5py.File(filename, "r") as f:
    f.visititems(print_hdf5_structure)
    print(f["data/demo_5/obs/ee_pos"][0])
    print(f["data/demo_5/obs/ee_states"][0])
    print(f["data/demo_5/obs/gripper_states"][0])

    # print(f["data/demo_5/obs/joint_pos"][0])

    # print(f["data/demo_5/obs/robot_state"][0])
    print(f["data/demo_5/states"][0])
    # print(f["data/demo_5/obs/states"][0])


#     # if mask.ndim == 3 and mask.shape[-1] == 3:
#     #     mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
#     # if mask.shape[:2] != image.shape[:2]:
#     #     mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
# import os
# import h5py
# import re

# # Path to the folder containing your HDF5 files
# folder = "/home/ubuntu/mount-point/libero_regenerate_retry/libero1/libero_object_reg"

# # Regex to match demo groups like "data/demo_45"
# demo_pattern = re.compile(r"^data/demo_\d+$")

# total_demos = 0

# # Loop through each .hdf5 file in the folder
# for filename in sorted(os.listdir(folder)):
#     if not filename.endswith(".hdf5"):
#         continue

#     filepath = os.path.join(folder, filename)
#     demo_count = 0

#     with h5py.File(filepath, "r") as f:
#         # Define a visitor function (no nonlocal needed)
#         def count_demos(name, obj):
#             if isinstance(obj, h5py.Group) and demo_pattern.match(name):
#                 # Use a global counter here
#                 nonlocal_demo_count[0] += 1

#         # workaround since Python disallows nonlocal at top level
#         nonlocal_demo_count = [0]
#         f.visititems(count_demos)
#         demo_count = nonlocal_demo_count[0]

#     total_demos += demo_count
#     print(f"{filename}: {demo_count} demos")

# print(f"\nTotal demos across all files: {total_demos}")
