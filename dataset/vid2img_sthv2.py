import os

VIDEO_ROOT = 'something-something-v2/20bn-something-something-v2'
FRAME_ROOT = 'something-something-v2/20bn-something-something-v2-frames'


def extract(video, tmpl='%06d.jpg'):
    video_name = video[:-5]
    output_dir = os.path.join(FRAME_ROOT, video_name)
    os.makedirs(output_dir, exist_ok=True)

    cmd = 'ffmpeg -i "{}/{}" -threads 1 -vf scale=-1:256 -q:v 0 "{}/{}"'.format(
        VIDEO_ROOT, video, output_dir, tmpl)
    os.system(cmd)
    print(f"视频 {video} 转换完成")


if __name__ == '__main__':
    if not os.path.exists(VIDEO_ROOT):
        raise ValueError('请检查视频目录是否存在')
    if not os.path.exists(FRAME_ROOT):
        os.makedirs(FRAME_ROOT)

    # 生成视频1到100的文件名（非格式化）
    for i in range(1, 101):
        video_name = f"{i}.webm"  # 直接使用 i 作为文件名（如1.webm, 10.webm）
        video_path = os.path.join(VIDEO_ROOT, video_name)

        if os.path.exists(video_path):
            extract(video_name)
        else:
            print(f"文件不存在: {video_name}，跳过")