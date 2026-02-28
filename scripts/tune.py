from ultralytics import YOLO
import torch
import gc
import os

# --- CẤU HÌNH ---
DATA_YAML = 'configs/rip_current_seg.yaml' 

# Danh sách model (Ưu tiên YOLO11n-seg cho Jetson Nano)
MODELS_TO_TUNE = [
    'weights/yolo11n-seg.pt',
    # 'weights/yolov8n-seg.pt', 
]

def run_tuning():
    # Kiểm tra thiết bị
    if torch.cuda.is_available():
        device = 0
        device_name = torch.cuda.get_device_name(0)
    else:
        device = 'cpu'
        device_name = "CPU"
        
    print(f"--- BẮT ĐẦU TUNING TRÊN THIẾT BỊ: {device_name} ---")
    print(f"⚠️ Lưu ý: Tuning tốn rất nhiều thời gian. Hãy kiên nhẫn!")
    
    for model_name in MODELS_TO_TUNE:
        print(f"\n{'-'*60}")
        print(f"🛠️ ĐANG TUNE MODEL: {model_name}")
        print(f"{'-'*60}")
        
        try:
            # 1. Giải phóng bộ nhớ triệt để trước khi bắt đầu
            torch.cuda.empty_cache()
            gc.collect()

            model = YOLO(model_name)

            # Cấu hình Tuning
            tune_args = {
                'data': DATA_YAML,
                'epochs': 10,
                'iterations': 30,
                'optimizer': 'AdamW',
                'val': True,
                'plots': False,
                'save': False,
                'imgsz': 640,
                'batch': 16,
                'device': device,
                'workers': 4,
                'project': 'RipCurrent_Tuning',
                'name': f'tune_{model_name.replace("weights/", "").replace(".pt", "")}',
            }
            
            # Tự động chọn task chính xác để tránh lỗi argument
            if '-seg' in model_name or 'FastSAM' in model_name:
                tune_args['task'] = 'segment'
                print("👉 Mode Tuning: SEGMENTATION")
            else:
                tune_args['task'] = 'detect'
                print("👉 Mode Tuning: DETECTION")

            # Chúng ta tune dựa trên việc tối đa hóa mAP trên tập validation
            model.tune(**tune_args)
            
        except Exception as e:
            print(f"❌ Lỗi khi tune {model_name}: {e}")
            if "CUDA out of memory" in str(e):
                print("👉 Gợi ý: Hãy giảm batch=8 hoặc imgsz=512 trong code.")
            continue
    
    print("\n" + "="*60)
    print("✅ QUÁ TRÌNH TUNING HOÀN TẤT!")
    print("👉 Bước tiếp theo: Vào thư mục 'RipCurrent_Tuning/tune_.../weights/'")
    print("👉 Tìm file 'best_hyperparameters.yaml' để lấy thông số train.")
    print("="*60)

if __name__ == '__main__':
    run_tuning()