import os
import shutil
import glob
import yaml
from sklearn.model_selection import train_test_split

# --- CẤU HÌNH ĐƯỜNG DẪN (Dựa trên ảnh của Vic) ---
RAW_IMG_DIR = 'data/RipDetSeg_v1.1.6_train/train_images'
# [QUAN TRỌNG] Đổi sang thư mục chứa nhãn Segmentatiodn
RAW_LABEL_DIR = 'data/RipDetSeg_v1.1.6_train/train_labels_segmentation' 

OUTPUT_DIR = 'datasets'

def setup_data():
    # 1. Tạo thư mục
    for split in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'labels', split), exist_ok=True)

    # 2. Lấy danh sách file
    image_files = glob.glob(os.path.join(RAW_IMG_DIR, '*.*'))
    valid_pairs = []

    print(f"Dataset gốc: {RAW_IMG_DIR}")
    print(f"Nhãn Segmentation: {RAW_LABEL_DIR}")
    
    for img_path in image_files:
        filename = os.path.basename(img_path)
        name_no_ext = os.path.splitext(filename)[0]
        
        # Tìm file txt trong folder segmentation
        label_path = os.path.join(RAW_LABEL_DIR, name_no_ext + '.txt')
        
        if os.path.exists(label_path):
            valid_pairs.append((img_path, label_path))
    
    print(f">>> Tìm thấy {len(valid_pairs)} cặp ảnh/nhãn Segmentation hợp lệ.")

    # 3. Chia tập dữ liệu 80/20
    train_pairs, val_pairs = train_test_split(valid_pairs, test_size=0.2, random_state=42)

    def copy_files(pairs, split):
        print(f"Copying {len(pairs)} files to {split}...")
        for img_src, lbl_src in pairs:
            shutil.copy2(img_src, os.path.join(OUTPUT_DIR, 'images', split, os.path.basename(img_src)))
            shutil.copy2(lbl_src, os.path.join(OUTPUT_DIR, 'labels', split, os.path.basename(lbl_src)))

    copy_files(train_pairs, 'train')
    copy_files(val_pairs, 'val')

    # 4. Tạo file YAML cấu hình
    yaml_content = {
        'path': os.path.abspath(OUTPUT_DIR),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 1,
        'names': ['rip_current']
    }

    with open('configs/rip_current_seg.yaml', 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print("\n✅ Đã tạo file cấu hình: rip_current_seg.yaml")

if __name__ == '__main__':
    setup_data()