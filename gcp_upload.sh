# to upload folder in ~/path/folder
# cd ~/path
# gsutil -m cp -r folder/ gs://sereact_lerobot_data/
# can delete "folder" in ~/path/ after uploading finished
cd /home/ubuntu/mount-point/libero_regenerate_retry/libero1/
gsutil -m cp -r libero_object_reg/* gs://sereact_lerobot_data/libero_regenerate/
# gsutil -m cp -r h5_tienkung_xsens_1rgb/* gs://sereact_lerobot_data/robomind/benchmark1_0_compressed/h5_tienkung_xsens_1rgb

# gsutil -m cp -r h5_franka_3rgb/* gs://sereact_lerobot_data/robomind/benchmark1_0_compressed/h5_franka_3rgb