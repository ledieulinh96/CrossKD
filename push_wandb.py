import json
import wandb

# Initialize wandb project
wandb.init(project="KD", name="bin_12")

# Path to your file
#file_path = "/home/icns/linh/CrossKD/work_dirs/crosskd_r18_gflv1_r50_fpn_1x_coco_jsdbin/20241118_092639/vis_data/20241118_092639.json"  # Replace with your file path
file_path = "/home/icns/linh/CrossKD/work_dirs/crosskd_r18_gflv1_r50_fpn_1x_coco_jsdbin/20241122_014513/vis_data/20241122_014513.json"  # Replace with your file path

# Open the file and process line by line
with open(file_path, 'r') as file:
    for line in file:
        # Parse each line as JSON
        log_data = json.loads(line)
        
        # Log the data to wandb
        wandb.log(log_data)

# Finish the wandb session
wandb.finish()
