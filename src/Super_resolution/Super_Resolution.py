import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan_utils import RealESRGANer
import torch.multiprocessing as mp

def process_video(video_filename, input_folder, output_folder, device_id):
    torch.cuda.set_device(device_id)
    
    device = torch.device(f'cuda:{device_id}')
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    model_path = '/root/raw_mmau/Real-ESRGAN/weights/RealESRGAN_x4plus.pth' 
    upscaler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=True,
        device=device
    )
    
    input_video_path = os.path.join(input_folder, video_filename)
    output_video_path = os.path.join(output_folder, video_filename)

    print(f"Processing video: {video_filename} on GPU {device_id}")
    print(f"Input path: {input_video_path}")
    print(f"Output path: {output_video_path}")

 
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {input_video_path}")
        return

  
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video properties: width={frame_width}, height={frame_height}, fps={fps}, total_frames={total_frames}")

 
    scale_factor = min(max_width / (frame_width * 4), max_height / (frame_height * 4), 1)


    target_width = int(frame_width * 4 * scale_factor)
    target_height = int(frame_height * 4 * scale_factor)

  
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (target_width, target_height))

    if not out.isOpened():
        print(f"Error: Cannot open video writer for {output_video_path}")
        return

  
    for i in tqdm(range(total_frames), desc=f"Processing {video_filename}"):
        ret, frame = cap.read()
        if not ret:
            print(f"Frame {i} could not be read.")
            break

  
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        sr_frame, _ = upscaler.enhance(frame_rgb, outscale=4)

       
        sr_frame = np.clip(sr_frame, 0, 255).astype(np.uint8)

      
        sr_frame = cv2.cvtColor(sr_frame, cv2.COLOR_RGB2BGR)

  
        sr_frame_resized = cv2.resize(sr_frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

     
        out.write(sr_frame_resized)

    cap.release()
    out.release()

    output_file_size = os.path.getsize(output_video_path)
    print(f"生成的视频文件 {output_video_path} 大小为: {output_file_size} 字节")


def worker(gpu_id, video_filenames, input_folder, output_folder):
    for video_filename in video_filenames:
        process_video(video_filename, input_folder, output_folder, gpu_id)


if __name__ == '__main__':

    input_folder = '/root/raw_mmau/traindd/'
    output_folder = '/root/raw_mmau/traindp'
    
 
    os.makedirs(output_folder, exist_ok=True)

    video_filenames = sorted([f for f in os.listdir(input_folder) if f.endswith('.mp4')])


    max_width = 3840 
    max_height = 2160  


    start_file = '013.mp4'
    start_index = video_filenames.index(start_file)

    video_filenames = video_filenames[start_index:]

    num_gpus = 4
    for i in range(0, len(video_filenames), num_gpus):
        video_chunk = video_filenames[i:i + num_gpus]
        processes = []
        for gpu_id, video_filename in enumerate(video_chunk):
            p = mp.Process(target=worker, args=(gpu_id, [video_filename], input_folder, output_folder))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    print("所有视频增强完成！")
