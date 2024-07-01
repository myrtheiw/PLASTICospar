import ultralytics
from ultralytics import YOLO
import cv2
import os
import csv
from ultralytics.trackers.bot_sort import BOTSORT

# Define paths
image_folder = '/home/myrtheiw/PlasticNoria/Data' 
output_folder = '/home/myrtheiw/PlasticNoria/Data_output'  
model_path = '/home/myrtheiw/PlasticNoria/PLASTIC/FixModel/yolov8s/weights/last.pt'

# Ensure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Load YOLO model
model = YOLO(model_path)

# Initialize CSV writer
csv_file_path = os.path.join(output_folder, "results.csv")
csv_columns = ['Image', 'Class', 'Count']
csv_data = []

args = {
    'proximity_thresh': 0.5,
    'appearance_thresh': 0.25,
    'frame_rate': 30,
    'track_buffer': 50
}

# Initialize tracker
tracker = BOTSORT(args=args, frame_rate=args['frame_rate'])

# Process each image in the folder
for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(image_folder, filename)
        output_image_path = os.path.join(output_folder, filename)

        # Read and process the image
        im0 = cv2.imread(image_path)
        if im0 is None:
            print(f"Error reading image {image_path}")
            continue

        # Detect objects
        results = model(im0)

        # Initialize a dictionary to count objects
        object_counts = {}

        # Draw bounding boxes and labels on the image
        for result in results:
            for bbox in result.boxes:
                cls_id = int(bbox.cls)
                label = model.names[cls_id]
                confidence = float(bbox.conf)
                xmin, ymin, xmax, ymax = bbox.xyxy[0]

                # Draw the bounding box and label on the image
                cv2.rectangle(im0, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
                cv2.putText(im0, f'{label} {confidence:.2f}', (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Count objects
                if label in object_counts:
                    object_counts[label] += 1
                else:
                    object_counts[label] = 1

        # Save the annotated image
        cv2.imwrite(output_image_path, im0)
        print(f"Processed and saved: {output_image_path}")

        # Prepare data for CSV
        for label, count in object_counts.items():
            csv_data.append({'Image': filename, 'Class': label, 'Count': count})

# Write CSV file
try:
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in csv_data:
            writer.writerow(data)
    print(f"Results exported to: {csv_file_path}")
except IOError:
    print("I/O error while writing CSV file")

cv2.destroyAllWindows()
