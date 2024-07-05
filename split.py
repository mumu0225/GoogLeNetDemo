import os
import shutil
import random


def split_dataset(source_dir, train_dir, val_dir, val_split=0.2):
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    for class_name in classes:
        class_dir = os.path.join(source_dir, class_name)
        images = os.listdir(class_dir)
        random.shuffle(images)

        val_size = int(len(images) * val_split)
        train_images = images[val_size:]
        val_images = images[:val_size]

        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)

        if not os.path.exists(train_class_dir):
            os.makedirs(train_class_dir)
        if not os.path.exists(val_class_dir):
            os.makedirs(val_class_dir)

        for image in train_images:
            src = os.path.join(class_dir, image)
            dst = os.path.join(train_class_dir, image)
            shutil.copyfile(src, dst)

        for image in val_images:
            src = os.path.join(class_dir, image)
            dst = os.path.join(val_class_dir, image)
            shutil.copyfile(src, dst)


source_dir = '101_ObjectCategories'
train_dir = 'Caltech 101/train'
val_dir = 'Caltech 101/val'
val_split = 0.2

split_dataset(source_dir, train_dir, val_dir, val_split)
