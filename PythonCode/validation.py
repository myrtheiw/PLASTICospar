from ultralytics import YOLO
import pandas as pd
import os


model_path = 'J:/Myrthe/PLASTICospar/FixModel/yolov8s/weights/last.pt'
data_path = 'J:/Myrthe/PLASTICospar/Dataset/OSPAR1.v1i.yolov8/data.yaml'
output_dir = 'J:/Myrthe/PLASTICospar/ValidationResults'

os.makedirs(output_dir, exist_ok=True)

model = YOLO(model_path)

validation_results = model.val(data=data_path)

results_df = pd.DataFrame(validation_results)
results_df['model_path'] = model_path

# Save to CSV
output_file = os.path.join(output_dir, 'validation_results.csv')
results_df.to_csv(output_file, index=False)

print("Validation results saved")