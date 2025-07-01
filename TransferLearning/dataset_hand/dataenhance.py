import os
import struct
import numpy as np
from PIL import Image, ImageEnhance
import random

image_roots = ["digits_stu1", "digits_stu2", "digits_stu3"]

train_img_path = "train-images-idx3-ubyte"
train_lbl_path = "train-labels-idx1-ubyte"
test_img_path = "t10k-images-idx3-ubyte"
test_lbl_path = "t10k-labels-idx1-ubyte"

images = []
labels = []

def augment_image(img, num_aug=15):
    """对单张图像进行 num_aug 次增强"""
    augmented = []
    for _ in range(num_aug):
        aug = img.copy()

        # 随机旋转
        angle = random.uniform(-15, 15)
        aug = aug.rotate(angle)

        # 随机平移
        dx = random.uniform(-3, 3)
        dy = random.uniform(-3, 3)
        aug = aug.transform(
            aug.size,
            Image.AFFINE,
            (1, 0, dx, 0, 1, dy),
            fillcolor=255
        )

        # 随机对比度
        contrast = ImageEnhance.Contrast(aug)
        aug = contrast.enhance(random.uniform(0.8, 1.2))

        # 随机亮度
        brightness = ImageEnhance.Brightness(aug)
        aug = brightness.enhance(random.uniform(0.8, 1.2))

        # 转换为 numpy
        aug_np = np.asarray(aug, dtype=np.uint8)
        augmented.append(aug_np)
    return augmented

print("加载并增强图像统计：")
for image_root in image_roots:
    root_image_count = 0
    print(f"\n处理目录: {image_root}")
    
    for label in sorted(os.listdir(image_root)):
        label_dir = os.path.join(image_root, label)
        if not os.path.isdir(label_dir) or not label.isdigit():
            continue

        label_image_count = 0
        for filename in sorted(os.listdir(label_dir)):
            if filename.lower().endswith(".png"):
                img_path = os.path.join(label_dir, filename)
                img = Image.open(img_path).convert('L').resize((28, 28))
                img_array = np.asarray(img, dtype=np.uint8)
                images.append(img_array)
                labels.append(int(label))

                # 增强
                aug_imgs = augment_image(img, num_aug=15)
                images.extend(aug_imgs)
                labels.extend([int(label)] * len(aug_imgs))
                label_image_count += 1 + len(aug_imgs)

        if label_image_count > 0:
            print(f"  🏷️ 类别 {label}: {label_image_count} 张图像")
            root_image_count += label_image_count

    print(f"总计加载自 {image_root}: {root_image_count} 张图像")

if not images:
    raise ValueError("没有找到任何图像，请检查所有 digits 文件夹是否存在图像文件。")

images = np.stack(images).reshape(-1, 28 * 28)
labels = np.array(labels, dtype=np.uint8)

indices = np.arange(len(images))
np.random.shuffle(indices)

split = int(0.8 * len(images))
train_idx, test_idx = indices[:split], indices[split:]

train_images = images[train_idx]
train_labels = labels[train_idx]
test_images = images[test_idx]
test_labels = labels[test_idx]

with open(train_img_path, 'wb') as f:
    f.write(struct.pack('>IIII', 2051, len(train_images), 28, 28))
    f.write(train_images.tobytes())

with open(train_lbl_path, 'wb') as f:
    f.write(struct.pack('>II', 2049, len(train_labels)))
    f.write(train_labels.tobytes())
    
with open(test_img_path, 'wb') as f:
    f.write(struct.pack('>IIII', 2051, len(test_images), 28, 28))
    f.write(test_images.tobytes())

with open(test_lbl_path, 'wb') as f:
    f.write(struct.pack('>II', 2049, len(test_labels)))
    f.write(test_labels.tobytes())

print(f"\n最终图像数: {len(images)}（训练 {len(train_images)}，测试 {len(test_images)}）")
print("文件已保存为标准 MNIST 格式：")
print(f"{train_img_path}")
print(f"{train_lbl_path}")
print(f"{test_img_path}")
print(f"{test_lbl_path}")
