import shutil
from pathlib import Path

# Đường dẫn gốc
root_dir = Path(r"D:\scr\journal\data\ViSa")

for sub_dir in root_dir.iterdir():
    if not sub_dir.is_dir():
        continue

    print(f"🟢 Đang xử lý: {sub_dir.name}")

    data_dir = sub_dir / "Data"
    normal_dir = data_dir / "Images" / "Normal"
    anomaly_dir = data_dir / "Images" / "Anomaly"

    # Kiểm tra tồn tại
    if not normal_dir.exists() or not anomaly_dir.exists():
        print(f"⚠️  Bỏ qua {sub_dir.name} (thiếu Normal/Anomaly)")
        continue

    # Tạo thư mục train/test cùng cấp với Data
    train_good = sub_dir / "train" / "good"
    test_good = sub_dir / "test" / "good"
    test_not_good = sub_dir / "test" / "not_good"
    for d in [train_good, test_good, test_not_good]:
        d.mkdir(parents=True, exist_ok=True)

    # ============================================
    # 1️⃣ Di chuyển ảnh NORMAL
    # ============================================
    normal_images = sorted(normal_dir.glob("*"))
    for img_path in normal_images:
        stem = img_path.stem
        try:
            num = int(stem[:3])  # Lấy 3 ký tự đầu tiên, vd: 000.png -> 0
        except ValueError:
            print(f"⚠️ Bỏ qua file không hợp lệ: {img_path.name}")
            continue

        if 0 <= num <= 50:
            dest = train_good / img_path.name
        elif 51 <= num <= 78:
            dest = test_good / img_path.name
        else:
            continue

        shutil.move(str(img_path), str(dest))

    # ============================================
    # 2️⃣ Di chuyển ảnh ANOMALY
    # ============================================
    for img_path in anomaly_dir.glob("*"):
        dest = test_not_good / img_path.name
        shutil.move(str(img_path), str(dest))

    # ============================================
    # 3️⃣ Xóa thư mục khác, giữ lại train/test
    # ============================================
    for child in sub_dir.iterdir():
        if child.name not in ["train", "test"]:
            try:
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()  # nếu là file
                print(f"🧹 Đã xóa: {child}")
            except Exception as e:
                print(f"❌ Lỗi khi xóa {child}: {e}")

    print(f"✅ Hoàn tất: {sub_dir.name}")

print("🎯 Tất cả hoàn tất, chỉ còn train/ và test/ trong mỗi thư mục!")
