from ultralytics import YOLO
import cv2
import numpy as np
import os
import glob
import time

# --- CẤU HÌNH ---
# 1. Model Detection (Nhanh)
MODEL_PATH = 'yolov8n.pt'  # Hoặc yolov5nu.pt
TEST_IMAGES_DIR = 'datasets/rip_current_seg/images/val'

# 2. Nơi lưu dữ liệu Segmentation đầu ra
SAVE_DIR = 'saved_segmentation_data'
SAVE_IMG_DIR = os.path.join(SAVE_DIR, 'images')
SAVE_MASK_DIR = os.path.join(SAVE_DIR, 'masks') # Lưu ảnh trắng đen

def create_mask_from_box(img_shape, boxes, sigma_scale=0.3, threshold=0.4):
    """
    Tạo Mask nhị phân từ Bounding Box thông qua Gaussian Heatmap
    """
    h, w = img_shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)
    
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        
        # Tạo lưới Gaussian trong vùng box
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        width = x2 - x1
        height = y2 - y1
        
        sigma_x = width * sigma_scale
        sigma_y = height * sigma_scale
        
        if sigma_x == 0 or sigma_y == 0: continue
            
        y_grid, x_grid = np.ogrid[y1:y2, x1:x2]
        blob = np.exp(-((x_grid - center_x)**2 / (2 * sigma_x**2) + 
                        (y_grid - center_y)**2 / (2 * sigma_y**2)))
        
        # Cộng dồn heatmap
        heatmap[y1:y2, x1:x2] = np.maximum(heatmap[y1:y2, x1:x2], blob)

    # --- CHUYỂN ĐỔI THÀNH MASK NHỊ PHÂN ---
    # Những điểm có độ nóng > threshold (ví dụ 40%) sẽ thành màu trắng (255)
    # Còn lại là đen (0)
    binary_mask = (heatmap > threshold).astype(np.uint8) * 255
    
    return binary_mask

def run_conversion():
    # Tạo thư mục lưu
    os.makedirs(SAVE_IMG_DIR, exist_ok=True)
    os.makedirs(SAVE_MASK_DIR, exist_ok=True)
    
    print(f"Loading Model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    img_paths = glob.glob(os.path.join(TEST_IMAGES_DIR, '*.jpg'))[:10]
    print(f"Đang xử lý {len(img_paths)} ảnh...")
    
    for img_path in img_paths:
        filename = os.path.basename(img_path)
        
        # 1. Detect bằng YOLO
        results = model(img_path, verbose=False)[0]
        
        if len(results.boxes) > 0:
            orig_img = cv2.imread(img_path)
            boxes = results.boxes.xyxy.cpu().numpy()
            
            # 2. Tạo Mask giả lập từ Box
            mask = create_mask_from_box(orig_img.shape, boxes, sigma_scale=0.3, threshold=0.5)
            
            # 3. Lưu dữ liệu
            # Lưu ảnh gốc
            cv2.imwrite(os.path.join(SAVE_IMG_DIR, filename), orig_img)
            
            # Lưu Mask (dạng ảnh PNG trắng đen)
            # Định dạng này có thể dùng để train model Segmentation sau này!
            mask_filename = filename.replace('.jpg', '.png')
            cv2.imwrite(os.path.join(SAVE_MASK_DIR, mask_filename), mask)
            
            # (Tùy chọn) Hiển thị kết quả kiểm tra
            # Tạo viền xanh bao quanh mask đè lên ảnh gốc để Vic xem thử
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            preview_img = orig_img.copy()
            cv2.drawContours(preview_img, contours, -1, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(SAVE_DIR, f"preview_{filename}"), preview_img)
            
            print(f"✅ Saved mask for: {filename}")
        else:
            print(f"Skipped (No detection): {filename}")

if __name__ == '__main__':
    run_conversion()