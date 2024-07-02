from ultralytics import YOLO

model_path = '/home/myrtheiw/PLASTICospar/FixModel/yolov8s/weights/last.pt'
data_path = '/home/myrtheiw/PLASTICospar/Dataset/Ospar.v1i.yolov8/data.yaml'

model = YOLO(model_path)

validation_results = model.val(data=data_path)