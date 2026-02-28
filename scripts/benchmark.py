from ultralytics import YOLO
import glob
import os
import pandas as pd
import torch

PROJECT_DIR = 'RipCurrent_SOTA_Battle'
EXPORT_DIR = 'Models_SOTA_Jetson'

def benchmark_and_export():
    os.makedirs(EXPORT_DIR, exist_ok=True)
    
    # Tìm file best.pt
    model_paths = glob.glob(os.path.join(PROJECT_DIR, '*/weights/best.pt'))
    
    if not model_paths:
        print("❌ Chưa có model nào! Hãy chạy file train trước.")
        return

    results = []
    print(f"🔎 Tìm thấy {len(model_paths)} mô hình. Đang đánh giá & export...")

    for path in model_paths:
        model_name = path.split(os.sep)[-3].replace('train_', '') # Lấy tên gọn
        print(f"\n--- Xử lý: {model_name} ---")
        
        try:
            model = YOLO(path)
            
            # 1. ĐÁNH GIÁ (Benchmark)
            # Lấy mAP trên tập Validation
            metrics = model.val(data='configs/rip_current_seg.yaml', split='val', verbose=False)
            
            map50_mask = metrics.seg.map50
            map50_95_mask = metrics.seg.map
            
            # Đo số lượng tham số (Parameters) để ước lượng độ nặng
            params = sum(p.numel() for p in model.parameters()) / 1e6
            
            results.append({
                "Model": model_name,
                "mAP@50 (Mask)": round(map50_mask, 4),
                "mAP@50-95 (Mask)": round(map50_95_mask, 4),
                "Params (Triệu)": round(params, 2),
                "Exported": "Yes"
            })
            
            # 2. EXPORT CHO JETSON (ONNX)
            print(f"👉 Exporting {model_name} to ONNX...")
            model.export(format='onnx', opset=12, dynamic=False)
            
            # Di chuyển file ONNX ra thư mục chung
            src_onnx = path.replace('.pt', '.onnx')
            dst_onnx = os.path.join(EXPORT_DIR, f"{model_name}.onnx")
            
            if os.path.exists(src_onnx):
                if os.path.exists(dst_onnx): os.remove(dst_onnx)
                os.rename(src_onnx, dst_onnx)
        
        except Exception as e:
            print(f"⚠️ Lỗi xử lý {model_name}: {e}")
            results.append({"Model": model_name, "Exported": "Failed", "Error": str(e)})

    # --- XUẤT BẢNG SO SÁNH ---
    df = pd.DataFrame(results)
    # Sắp xếp theo độ chính xác giảm dần
    if not df.empty and "mAP@50-95 (Mask)" in df.columns:
        df = df.sort_values(by="mAP@50-95 (Mask)", ascending=False)
        
    print("\n🏆 BẢNG XẾP HẠNG HIỆU NĂNG SEGMENTATION 🏆")
    print(df.to_string())
    
    df.to_csv("SOTA_Comparison_Report.csv", index=False)
    print("\n✅ Đã lưu báo cáo tại: SOTA_Comparison_Report.csv")
    print(f"✅ Đã lưu file ONNX tại thư mục: {EXPORT_DIR}/")

if __name__ == '__main__':
    benchmark_and_export()