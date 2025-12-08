# make_train_test_txt.py
import json
import random
from pathlib import Path


def main():
    # 这里就是你的 data_dir（存放 hsi/mask 的那个 HeiPor_resized_256）
    data_dir = "/mnt/nvme1n1/bitmhsi/dataset/HeiPor_resized_256"
    root = Path(data_dir)
    hsi_dir = root / "hsi"

    if not hsi_dir.exists():
        raise RuntimeError(f"hsi 目录不存在: {hsi_dir}")

    # 1) 收集所有 image_name（文件名去掉 .npy）
    image_names = sorted(f.stem for f in hsi_dir.glob("*.npy"))
    if not image_names:
        raise RuntimeError(f"hsi 目录下没有 .npy 文件: {hsi_dir}")

    print(f"总样本数: {len(image_names)}")

    # 2) 按 subject 分组：subject = image_name.split('#', 1)[0]
    subject_to_images = {}
    for name in image_names:
        subject = name.split("#", 1)[0]
        subject_to_images.setdefault(subject, []).append(name)

    subjects = sorted(subject_to_images.keys())
    print(f"总 subject 数(猪数): {len(subjects)}")

    # 3) 随机打乱 subject，按“8 只猪 train，3 只猪 test”划分
    rng = random.Random(2025)  # 固定随机种子，保证可复现
    subjects_shuffled = subjects[:]  # 拷贝一份
    rng.shuffle(subjects_shuffled)

    desired_train_subjects = 8
    total_subjects = len(subjects_shuffled)

    if total_subjects <= desired_train_subjects:
        # 如果猪总数比 8 少，就留 1 只给 test，其它都 train
        num_train_subjects = max(total_subjects - 1, 1)
    else:
        num_train_subjects = desired_train_subjects

    train_subjects = set(subjects_shuffled[:num_train_subjects])
    test_subjects  = set(subjects_shuffled[num_train_subjects:])

    print(f"Train subjects(猪): {len(train_subjects)} -> {sorted(train_subjects)}")
    print(f"Test  subjects(猪): {len(test_subjects)}  -> {sorted(test_subjects)}")

    # 4) 展开成具体 image_name 列表
    train_images = []
    test_images = []

    for subj, imgs in subject_to_images.items():
        if subj in train_subjects:
            train_images.extend(imgs)
        else:
            test_images.extend(imgs)

    train_images = sorted(train_images)
    test_images  = sorted(test_images)

    print(f"Train 样本数: {len(train_images)}")
    print(f"Test  样本数: {len(test_images)}")

    # 5) 写出到 txt，每行一个 image_name（不带 .npy）
    train_txt = root / "train.txt"
    test_txt  = root / "test.txt"

    with train_txt.open("w", encoding="utf-8") as f:
        for name in train_images:
            f.write(name + "\n")

    with test_txt.open("w", encoding="utf-8") as f:
        for name in test_images:
            f.write(name + "\n")

    # 6) 存一点划分信息，方便以后查
    split_info = {
        "random_seed": 2025,
        "num_total_images": len(image_names),
        "num_total_subjects": len(subjects),
        "num_train_images": len(train_images),
        "num_test_images": len(test_images),
        "num_train_subjects": len(train_subjects),
        "num_test_subjects": len(test_subjects),
        "train_subjects": sorted(train_subjects),
        "test_subjects": sorted(test_subjects),
    }

    with (root / "split_info.json").open("w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)

    print("\n划分完成！")
    print(f"train.txt: {train_txt}")
    print(f"test.txt : {test_txt}")
    print(f"split_info.json: {root / 'split_info.json'}")


if __name__ == "__main__":
    main()
