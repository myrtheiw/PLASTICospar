# tracker
import ultralytics
from ultralytics import YOLO
import cv2
import os
import csv
from collections import defaultdict

use_camera = False  #change to True is using camera, change to False if not using camera
camera_id = 0  # Change to the ID of the camera
video_path = '/home/myrtheiw/PLASTICospar/Dataset/Video/dirty1-16-39.mp4'
model_path = '/home/myrtheiw/PLASTICospar/FixModel/yolov8s/weights/last.pt'
output_video_path = '/home/myrtheiw/PLASTICospar/Dataset/Video/Dataset/OutputVideo/output_video.avi'
output_folder = '/home/myrtheiw/PLASTICospar/Dataset/Output_folder'



# Stride is the number of skipping in the video frame
stride = 1  # Process every 'stride' frames
frame_count = 0  # Initialize frame counter

# only change the three above
video_filename = os.path.basename(video_path)
video_name, video_ext = os.path.splitext(video_filename)

output_csv_path = os.path.join(output_folder, "Excel", video_name + ".csv")
print(output_csv_path)
output_video_path = os.path.join(output_folder, "Video", video_name + "_detected.avi")
print(output_video_path)

# Initialize the model
model = YOLO(model_path)

# Initialize a dictionary to count objects
object_counts = defaultdict(int)

# Process each frame in the video
results = model.track(source=video_path, show=True, tracker='bytetrack.yaml')

for result in results:
    if result is not None:
        for detection in result.detections:
            cls_id = int(detection.cls)
            label = model.names[cls_id]
            object_counts[label] += 1

# Write the counts to a CSV file
with open(output_csv_path, 'w', newline='') as csvfile:
    fieldnames = ['Class', 'Count']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for label, count in object_counts.items():
        writer.writerow({'Class': label, 'Count': count})

print(f"Results exported to: {output_csv_path}")