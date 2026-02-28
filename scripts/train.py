from ultralytics import YOLO
import torch
import gc
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# --- CẤU HÌNH ---
DATA_YAML = 'configs/rip_current_seg.yaml' 
IMG_SIZE = 640
EPOCHS = 50 
PROJECT_NAME = 'RipCurrent_Final_Arena'

# --- DANH SÁCH MODEL ---
MODELS_TO_COMPARE = [
    'weights/yolov5nu.pt',      # Nhóm Tốc độ
    'weights/yolov8s-seg.pt',   # Nhóm Ổn định
    'weights/yolo11n-seg.pt',   # Nhóm SOTA Nhẹ
    'weights/yolo11m-seg.pt',   # Nhóm Chính xác cao
    'weights/rtdetr-l.pt',      # Nhóm Transformer
]

def plot_and_save_metrics(project_dir):
    """
    Hàm này sẽ đi vào từng folder kết quả, đọc file results.csv
    và vẽ biểu đồ so sánh Loss, Recall giữa các model.
    """
    print(f"\n📊 ĐANG TỔNG HỢP DỮ LIỆU TỪ: {project_dir}...")
    
    # Tìm tất cả file results.csv
    csv_files = glob.glob(os.path.join(project_dir, '*/results.csv'))
    
    if not csv_files:
        print("⚠️ Không tìm thấy file kết quả nào để vẽ biểu đồ!")
        return

    # Tạo DataFrame tổng hợp
    summary_data = []
    
    plt.figure(figsize=(12, 10))
    
    # Chuẩn bị 2 biểu đồ con
    ax1 = plt.subplot(2, 1, 1) # Biểu đồ Loss
    ax2 = plt.subplot(2, 1, 2) # Biểu đồ Recall
    
    for file in csv_files:
        # Lấy tên model từ tên thư mục cha
        model_name = file.split(os.sep)[-2].replace('train_', '')
        
        try:
            # Đọc file CSV (Ultralytics csv thường có khoảng trắng ở tên cột, cần strip)
            df = pd.read_csv(file)
            df.columns = [c.strip() for c in df.columns] # Xóa khoảng trắng thừa
            
            # --- XỬ LÝ SỐ LIỆU ---
            epochs = df['epoch']
            
            # 1. Tổng hợp Loss (Box + Seg + Cls)
            # Tùy model mà cột có thể khác nhau (Detection ko có seg_loss)
            val_loss = df['val/box_loss'] # Bắt buộc có
            if 'val/seg_loss' in df.columns:
                val_loss += df['val/seg_loss']
            if 'val/cls_loss' in df.columns:
                val_loss += df['val/cls_loss']
                
            # 2. Lấy Recall (Ưu tiên Recall Mask nếu có, không thì lấy Box)
            if 'metrics/recall(M)' in df.columns:
                recall = df['metrics/recall(M)']
                metric_type = "(Mask)"
            else:
                recall = df['metrics/recall(B)']
                metric_type = "(Box)"
            
            # Vẽ lên biểu đồ
            ax1.plot(epochs, val_loss, label=f"{model_name}")
            ax2.plot(epochs, recall, label=f"{model_name} {metric_type}")
            
            # Lưu thông số tốt nhất vào bảng tổng hợp
            best_epoch_idx = recall.idxmax()
            summary_data.append({
                "Model": model_name,
                "Best Recall": recall.max(),
                "Final Val Loss": val_loss.iloc[-1],
                "Epoch đạt đỉnh": epochs[best_epoch_idx],
                "Link Log": file
            })
            
        except Exception as e:
            print(f"⚠️ Lỗi đọc file {file}: {e}")

    # Trang trí biểu đồ
    ax1.set_title("So sánh Validation Loss (Càng thấp càng tốt)")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Total Loss")
    ax1.legend()
    ax1.grid(True)
    
    ax2.set_title("So sánh Recall (Độ nhạy - Càng cao càng tốt)")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Recall Score")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    save_path = "Comparison_Charts.png"
    plt.savefig(save_path)
    print(f"✅ Đã lưu biểu đồ so sánh: {save_path}")
    
    # Lưu file Excel tổng hợp
    df_sum = pd.DataFrame(summary_data)
    df_sum.to_csv("Final_Training_Summary.csv", index=False)
    print("✅ Đã lưu bảng số liệu tổng hợp: Final_Training_Summary.csv")


def train_final_arena():
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"🚀 THIẾT BỊ HUẤN LUYỆN: {torch.cuda.get_device_name(0) if device == 0 else 'CPU'}")
    
    for model_name in MODELS_TO_COMPARE:
        print(f"\n{'='*60}")
        print(f"🔥 ĐANG HUẤN LUYỆN: {model_name.upper()}")
        print(f"{'='*60}")
        
        try:
            torch.cuda.empty_cache()
            gc.collect()

            model = YOLO(model_name)
            safe_name = model_name.replace('.pt', '')
            
            # Cấu hình Batch size
            if any(x in model_name for x in ['rtdetr', 'm-seg', 'l-seg']):
                batch_size = 4
            elif 'c-seg' in model_name or 's-seg' in model_name:
                batch_size = 8
            else:
                batch_size = 16

            # Cấu hình tham số
            train_args = {
                'data': DATA_YAML,
                'epochs': EPOCHS,
                'imgsz': IMG_SIZE,
                'device': device,
                'project': PROJECT_NAME,
                'name': f"train_{safe_name}",
                'patience': 15,
                'batch': batch_size,
                'exist_ok': True,
                'degrees': 10.0, 'fliplr': 0.5, 'mosaic': 1.0,
            }

            # Tự động thêm tham số mask
            if '-seg' in model_name or 'FastSAM' in model_name:
                train_args.update({'box': 7.5, 'cls': 0.5, 'mask': 1.0})
                print("👉 Mode: SEGMENTATION")
            else:
                train_args.update({'box': 7.5, 'cls': 0.5})
                print("👉 Mode: DETECTION")

            model.train(**train_args)
            print(f"✅ XONG: {model_name}")

        except Exception as e:
            print(f"❌ LỖI {model_name}: {e}")
            continue

    # --- BƯỚC CUỐI: TỔNG HỢP DỮ LIỆU ---
    plot_and_save_metrics(PROJECT_NAME)

if __name__ == '__main__':
    train_final_arena()