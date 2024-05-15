import os
import cv2

def images_to_video(image_folder, output_video_path, fps=30):
    images = [img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]
    images.sort()  # 确保图片按顺序排序

    if not images:
        print(f"No images found in {image_folder}")
        return

    # 读取第一张图片以获取帧的尺寸
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)

    video.release()
    print(f"Video saved to {output_video_path}")

def convert_all_folders_to_videos(root_folder, output_folder, fps=30):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for subdir in os.listdir(root_folder):
        subdir_path = os.path.join(root_folder, subdir)
        images_folder_path = os.path.join(subdir_path, 'images')
        if os.path.isdir(images_folder_path):
            output_video_path = os.path.join(output_folder, f"{subdir}.mp4")
            images_to_video(images_folder_path, output_video_path, fps)

if __name__ == "__main__":
    # 设置根文件夹路径和输出文件夹路径
    root_folder = './total_raw'  # 修改为你的根文件夹路径
    output_folder = './total_mp4'

    # 调用函数将所有子文件夹中的图片转换为视频
    convert_all_folders_to_videos(root_folder, output_folder)
