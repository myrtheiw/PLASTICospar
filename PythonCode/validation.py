from ultralytics import YOLO
import pandas as pd
import os


model_path = 'J:/Myrthe/PLASTICospar/FixModel/yolov8s/weights/last.pt'
data_path = 'J:/Myrthe/PLASTICospar/Dataset/OSPAR1.v1i.yolov8/data.yaml'
output_dir = 'J:/Myrthe/PLASTICospar/ValidationResults'

os.makedirs(output_dir, exist_ok=True)

model = YOLO(model_path)

validation_results = model.val(data=data_path)

results_dict = {
    'class': [],
    'images': [],
    'instances': [],
    'box_p': [],
    'box_r': [],
    'box_map50': [],
    'box_map50_95': []
}

for cls in validation_results['classes']:
    results_dict['class'].append(cls['name'])
    results_dict['images'].append(cls['images'])
    results_dict['instances'].append(cls['instances'])
    results_dict['box_p'].append(cls['box']['P'])
    results_dict['box_r'].append(cls['box']['R'])
    results_dict['box_map50'].append(cls['box']['mAP50'])
    results_dict['box_map50_95'].append(cls['box']['mAP50-95'])

# Convert the dictionary to a DataFrame
results_df = pd.DataFrame(results_dict)

# Add model path to DataFrame
results_df['model_path'] = model_path

# Save to CSV
output_file = os.path.join(output_dir, 'validation_results.csv')
results_df.to_csv(output_file, index=False)

print(f"Validation results saved to '{output_file}'")
print(validation_results.results)
print(validation_results.metrics)
print(validation_results.speed)
