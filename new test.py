import cv2
import os


def display_images(folder_path):
    # 获取指定文件夹中的所有文件
    files = os.listdir(folder_path)

    for file in files:
        # 检查文件是否为图片（假设图片的扩展名为 .jpg）
        if file.endswith(".bmp"):
            # 构建图片的完整路径
            image_path = os.path.join(folder_path, file)

            # 读取图片
            image = cv2.imread(image_path)

            # 显示图片
            if image is not None:
                cv2.imshow(file, image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 请替换成你的文件夹路径
folder_path = "C:/Users/13620/Desktop/l"
display_images(folder_path)
