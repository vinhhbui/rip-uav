from ultralytics import YOLO
import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

# Đường dẫn model đã train xong (Vic sửa lại cho đúng path thực tế)
MODEL_PATH = 'weights/yolov8s-seg.pt'
TEST_IMAGES_DIR = 'datasets/rip_current_seg/images/val' # Lấy ảnh trong tập val để test
OUTPUT_DIR = 'heatmap_results'

def apply_heatmap(image, mask, alpha=0.6):
    """
    Hàm chồng lớp Heatmap lên ảnh gốc
    alpha: độ trong suốt (0.6 là mask đậm, ảnh gốc mờ)
    """
    # 1. Tạo Heatmap từ mask
    # Mask đầu ra là binary (0, 1), ta nhân lên 255
    heatmap_base = (mask * 255).astype(np.uint8)
    
    # Làm mờ mask để tạo hiệu ứng tỏa nhiệt (Gaussian Blur)
    heatmap_blur = cv2.GaussianBlur(heatmap_base, (25, 25), 0)
    
    # Áp dụng bản đồ màu (COLORMAP_JET: Xanh -> Đỏ, hoặc COLORMAP_HOT: Đen -> Đỏ -> Vàng)
    heatmap_color = cv2.applyColorMap(heatmap_blur, cv2.COLORMAP_JET)
    
    # 2. Chồng lên ảnh gốc
    # Resize heatmap cho khớp ảnh gốc (phòng trường hợp size lệch)
    if heatmap_color.shape[:2] != image.shape[:2]:
        heatmap_color = cv2.resize(heatmap_color, (image.shape[1], image.shape[0]))
        
    # Xử lý: Chỉ tô màu vào vùng có mask (Mask > 0), vùng nền giữ nguyên ảnh gốc
    # Hoặc blend toàn bộ
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)
    
    return overlay

def create_mask_from_box(img_shape, boxes, sigma_scale=0.3, threshold=0.1):
    """
    Tạo Mask từ Bounding Box thông qua Gaussian Heatmap giả lập
    """
    h, w = img_shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)
    
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        
        sigma_x = width * sigma_scale
        sigma_y = height * sigma_scale
        
        y_grid, x_grid = np.ogrid[y1:y2, x1:x2]
        blob = np.exp(-((x_grid - center_x)**2 / (2 * sigma_x**2) + 
                        (y_grid - center_y)**2 / (2 * sigma_y**2)))
        
        heatmap[y1:y2, x1:x2] = np.maximum(heatmap[y1:y2, x1:x2], blob)

    return (heatmap > threshold).astype(np.float32)

def run_heatmap_test():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load model
    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    # Lấy 5 ảnh ngẫu nhiên để test
    img_paths = glob.glob(os.path.join(TEST_IMAGES_DIR, '*.jpg'))[:10]
    
    if not img_paths:
        print("Không tìm thấy ảnh để test!")
        return

    print(f"Đang xử lý {len(img_paths)} ảnh...")

    for img_path in img_paths:
        # Predict
        results = model(img_path, verbose=False)
        result = results[0]
        
        # Đọc ảnh gốc
        orig_img = cv2.imread(img_path)
        
        # Tính FPS từ inference speed của Ultralytics (ms)
        inference_time_ms = result.speed.get('inference', 0) if hasattr(result, 'speed') else 0
        fps = 1000 / inference_time_ms if inference_time_ms > 0 else 0
        
        if result.masks is not None:
            # 1. Model Segmentation
            masks = result.masks.data.cpu().numpy() 
            combined_mask = np.max(masks, axis=0) 
            combined_mask = cv2.resize(combined_mask, (orig_img.shape[1], orig_img.shape[0]))
            
            heatmap_img = apply_heatmap(orig_img, combined_mask)
            # Ghi FPS lên ảnh
            cv2.putText(heatmap_img, f"FPS: {fps:.1f} (Seg)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            filename = os.path.basename(img_path)
            save_path = os.path.join(OUTPUT_DIR, f"heatmap_{filename}")
            cv2.imwrite(save_path, heatmap_img)
            print(f"🔥 Đã lưu Heatmap (Seg): {save_path}")
            
        elif result.boxes is not None and len(result.boxes) > 0:
            # 2. Model Detection (BBox only)
            boxes = result.boxes.xyxy.cpu().numpy()
            combined_mask = create_mask_from_box(orig_img.shape, boxes)
            
            heatmap_img = apply_heatmap(orig_img, combined_mask)
            # Ghi FPS lên ảnh
            cv2.putText(heatmap_img, f"FPS: {fps:.1f} (Det)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            filename = os.path.basename(img_path)
            save_path = os.path.join(OUTPUT_DIR, f"heatmap_{filename}")
            cv2.imwrite(save_path, heatmap_img)
            print(f"🔥 Đã lưu Heatmap (Det giả lập): {save_path}")
            
        else:
            print(f"Không tìm thấy đối tượng trong ảnh: {os.path.basename(img_path)}")

if __name__ == '__main__':
    run_heatmap_test()