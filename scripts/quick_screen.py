from ultralytics import YOLO
import torch
import time
import pandas as pd
import gc

# --- CẤU HÌNH ---
DATA_YAML = 'rip_current_seg.yaml'
IMG_SIZE = 640
PILOT_EPOCHS = 10  # Chỉ train 10 epoch (Thay vì 100)
BATCH_SIZE = 16    # Giữ batch cố định để so sánh công bằng VRAM

# Danh sách các ứng viên
CANDIDATES = [
    'yolov5nu.pt',      # Ứng viên 1: Tốc độ
    'yolov8n-seg.pt',   # Ứng viên 2: Ổn định
    'yolo11n-seg.pt',   # Ứng viên 3: Công nghệ mới
    'yolo11s-seg.pt',   # Ứng viên 4: Chất lượng cao hơn
    # 'yolo11m-seg.pt'  # (Bỏ comment nếu muốn thử vận may với model nặng)
]

def quick_screen():
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"🚀 BẮT ĐẦU SÀNG LỌC TRÊN: {torch.cuda.get_device_name(0) if device == 0 else 'CPU'}")
    
    report = []

    for model_name in CANDIDATES:
        print(f"\n{'='*50}")
        print(f"🕵️  ĐANG KHÁM NGHIỆM: {model_name}")
        
        try:
            # 1. KIỂM TRA TỐC ĐỘ (INFERENCE SPEED TEST)
            # Load model (chưa train cũng đo được tốc độ cấu trúc mạng)
            model = YOLO(model_name)
            
            # Warmup
            dummy_input = torch.zeros((1, 3, IMG_SIZE, IMG_SIZE)).to(device)
            if device != 'cpu': model.model.to(device)
            _ = model.predict(source=dummy_input, verbose=False) # Chạy nháp
            
            # Đo FPS (Chạy 50 lần)
            t_start = time.time()
            for _ in range(50):
                _ = model.predict(source=dummy_input, verbose=False)
            t_end = time.time()
            
            avg_time = (t_end - t_start) / 50
            fps = 1.0 / avg_time
            print(f"⚡ Tốc độ ước tính: {fps:.2f} FPS")

            # 2. TRAIN NHÁP (PILOT TRAINING)
            print(f"📉 Train thử {PILOT_EPOCHS} epoch để xem khả năng học...")
            
            # Xả RAM để train
            torch.cuda.empty_cache()
            gc.collect()
            
            # Train ngắn hạn
            results = model.train(
                data=DATA_YAML,
                epochs=PILOT_EPOCHS,
                imgsz=IMG_SIZE,
                device=device,
                project='RipCurrent_Screening',
                name=f"screen_{model_name.replace('.pt','')}",
                batch=BATCH_SIZE,
                plots=False,
                verbose=False # Tắt log dài dòng
            )
            
            # Lấy mAP cuối cùng của đợt train nháp
            # Lưu ý: metrics.seg.map50 là mAP mask tại IoU 0.5
            map50 = results.seg.map50
            
            print(f"🎯 Kết quả sau {PILOT_EPOCHS} epoch: mAP@50 = {map50:.4f}")
            
            report.append({
                "Model": model_name,
                "FPS (PC)": round(fps, 1),
                "mAP@50 (Early)": round(map50, 4),
                "Status": "OK"
            })
            
        except Exception as e:
            print(f"❌ Model {model_name} thất bại: {e}")
            report.append({
                "Model": model_name,
                "FPS (PC)": 0,
                "mAP@50 (Early)": 0,
                "Status": "Failed (Out of Memory?)"
            })

    # --- TỔNG KẾT ---
    df = pd.DataFrame(report)
    # Tính điểm tiềm năng: (mAP * FPS) / 10 (Công thức tự chế để cân bằng)
    df['Score'] = df['mAP@50 (Early)'] * df['FPS (PC)']
    df = df.sort_values(by='Score', ascending=False)
    
    print("\n🏆 BẢNG KẾT QUẢ SÀNG LỌC NHANH 🏆")
    print(df.to_string())
    print("\n👉 LỜI KHUYÊN:")
    print("- Chọn model có 'mAP@50 (Early)' tăng nhanh nhất.")
    print("- Lưu ý FPS trên Jetson Nano sẽ thấp hơn trên PC khoảng 5-10 lần.")
    print("- Chỉ cần train FULL (100 epoch) cho Top 1 và Top 2 của bảng này.")

if __name__ == '__main__':
    quick_screen()