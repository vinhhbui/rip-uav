from ultralytics import YOLO

MODEL_PATH = 'RipCurrent_Seg_Project/train_seg_v8n/weights/best.pt'

def export():
    model = YOLO(MODEL_PATH)
    print("Đang export model Segmentation sang ONNX...")
    # opset=12 ổn định nhất cho Jetson
    model.export(format='onnx', opset=12, dynamic=False)
    print("✅ Export xong. Hãy copy file .onnx vào Jetson Nano.")

if __name__ == '__main__':
    export()