import os
import cv2
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RemotePPG:
    def __init__(self, buffer_size=150, sampling_rate=30):
        """
        初始化rPPG类

        参数:
            buffer_size: 信号缓冲区大小
            sampling_rate: 采样率(Hz)
        """

        self.buffer_size = buffer_size
        self.fs = sampling_rate

        # 初始化缓冲区
        self.forehead_buffer = []
        self.cheek_buffer = []
        self.timestamps = []

        # 加载人脸检测器
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect_faces(self, frame):
        """
        检测图像中的人脸

        参数:
            frame: 输入图像

        返回:
            faces: 人脸位置列表
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(120, 120)  # 这个与人与摄像头的距离有关需要根据实际情况调整
        )
        return faces

    def extract_face_roi(self, frame, faces):
        """
        从人脸中提取感兴趣区域(ROI)

        参数:
            frame: 输入图像
            faces: 人脸位置列表

        返回:
            包含ROI数据和显示图像的字典
        """
        if len(faces) == 0:
            return None

        # 选择最大的人脸
        face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = face

        # 创建显示图像副本
        display = frame.copy()

        # 绘制人脸框
        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 计算前额ROI (上部1/3)
        forehead_y = y + int(h * 0.1)
        forehead_h = int(h * 0.25)
        forehead_roi = frame[forehead_y:forehead_y + forehead_h, x:x + w]

        # 计算脸颊ROI (中间部分)
        cheek_y = y + int(h * 0.4)
        cheek_h = int(h * 0.25)
        cheek_roi = frame[cheek_y:cheek_y + cheek_h, x:x + w]

        # 在显示图像上标记ROI
        cv2.rectangle(display, (x, forehead_y), (x + w, forehead_y + forehead_h), (255, 0, 0), 2)
        cv2.rectangle(display, (x, cheek_y), (x + w, cheek_y + cheek_h), (0, 0, 255), 2)

        # 添加标签
        cv2.putText(display, "Forehead", (x, forehead_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
        cv2.putText(display, "Cheek", (x, cheek_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

        return {
            'forehead': forehead_roi,
            'cheek': cheek_roi,
            'face': (x, y, w, h),
            'display': display
        }

    def process_roi(self, roi):
        """
        处理ROI区域，提取平均RGB值

        参数:
            roi: ROI图像区域

        返回:
            RGB平均值元组
        """
        if roi is None or roi.size == 0:
            return None

        # 计算ROI区域的平均RGB值
        mean_rgb = cv2.mean(roi)[:3]  # 仅取RGB通道

        return mean_rgb


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def calculate_heart_rate(signal, fs):
    # 使用FFT计算频谱
    fft_result = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), d=1 / fs)

    # 找到最大峰值对应的频率
    positive_freqs = freqs[:len(freqs) // 2]
    magnitudes = np.abs(fft_result[:len(freqs) // 2])
    peak_index = np.argmax(magnitudes)
    heart_rate = positive_freqs[peak_index] * 60

    return heart_rate


def main():
    # 设置工作目录为脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # 直接指定输入和输出路径
    input_path = "yzw515.mp4"
    output_path = "processed_yyy.mp4"

    # 创建RemotePPG对象
    rppg = RemotePPG(buffer_size=150, sampling_rate=30)

    # 初始化视频捕获
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        logger.error(f"无法打开视频: {input_path}")
        return

    # 获取视频参数
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 更新采样率
    rppg.fs = fps

    logger.info(f"视频信息: {frame_width}x{frame_height}, {fps} FPS, {total_frames} 帧")

    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # 存储RGB值
    forehead_rgb_values = {'R': [], 'G': [], 'B': []}
    cheek_rgb_values = {'R': [], 'G': [], 'B': []}

    try:
        # 进度跟踪
        start_time = time.time()
        frame_idx = 0

        while True:
            # 读取一帧
            ret, frame = cap.read()

            if not ret:
                break

            frame_idx += 1

            # 检测人脸
            faces = rppg.detect_faces(frame)

            # 提取ROI并绘制标记
            result = rppg.extract_face_roi(frame, faces)

            if result is not None:
                mean_rgb_forehead = rppg.process_roi(result['forehead'])
                mean_rgb_cheek = rppg.process_roi(result['cheek'])

                # 打印RGB平均值
                logger.info(f"Frame {frame_idx}: Forehead RGB: {mean_rgb_forehead}, Cheek RGB: {mean_rgb_cheek}")

                # 将BGR图像转换为RGB以便matplotlib正确显示
                display_rgb = cv2.cvtColor(result['display'], cv2.COLOR_BGR2RGB)

                # 显示结果图像
                cv2.imshow('处理中...', result['display'])

                # 保存结果帧到输出视频
                video_writer.write(result['display'])

                # 记录RGB值
                if mean_rgb_forehead is not None:
                    forehead_rgb_values['R'].append(mean_rgb_forehead[2])
                    forehead_rgb_values['G'].append(mean_rgb_forehead[1])
                    forehead_rgb_values['B'].append(mean_rgb_forehead[0])

                if mean_rgb_cheek is not None:
                    cheek_rgb_values['R'].append(mean_rgb_cheek[2])
                    cheek_rgb_values['G'].append(mean_rgb_cheek[1])
                    cheek_rgb_values['B'].append(mean_rgb_cheek[0])
            else:
                # 如果没有检测到人脸，直接保存原帧
                cv2.imshow('处理中...', frame)
                video_writer.write(frame)

            # 显示进度
            if frame_idx % 30 == 0:
                elapsed = time.time() - start_time
                fps_avg = frame_idx / elapsed if elapsed > 0 else 0
                progress = frame_idx / total_frames * 100 if total_frames > 0 else 0
                eta = (total_frames - frame_idx) / fps_avg if fps_avg > 0 else 0

                logger.info(f"进度: {progress:.1f}% ({frame_idx}/{total_frames}), "
                            f"FPS: {fps_avg:.1f}, ETA: {eta:.1f}秒")

            # 按'q'退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()

        logger.info(f"视频处理完成，共处理 {frame_idx} 帧")

        # 绘制RGB变化曲线
        plot_rgb_changes(forehead_rgb_values, cheek_rgb_values)

        # 计算并绘制BVP信号
        compute_and_plot_bvp(forehead_rgb_values, fps)


def plot_rgb_changes(forehead_rgb_values, cheek_rgb_values):
    """
    绘制前额和脸颊的RGB变化曲线

    参数:
        forehead_rgb_values: 前额的RGB值字典
        cheek_rgb_values: 腮红的RGB值字典
    """
    frames = range(len(forehead_rgb_values['R']))

    plt.figure(figsize=(12, 8))

    # 绘制前额的RGB变化曲线
    plt.subplot(2, 1, 1)
    plt.plot(frames, forehead_rgb_values['R'], label='Forehead R', color='red')
    plt.plot(frames, forehead_rgb_values['G'], label='Forehead G', color='green')
    plt.plot(frames, forehead_rgb_values['B'], label='Forehead B', color='blue')
    plt.title('Forehead RGB Values Over Time')
    plt.xlabel('Frame Number')
    plt.ylabel('Average RGB Value')
    plt.legend()

    # 绘制脸颊的RGB变化曲线
    plt.subplot(2, 1, 2)
    plt.plot(frames, cheek_rgb_values['R'], label='Cheek R', color='red')
    plt.plot(frames, cheek_rgb_values['G'], label='Cheek G', color='green')
    plt.plot(frames, cheek_rgb_values['B'], label='Cheek B', color='blue')
    plt.title('Cheek RGB Values Over Time')
    plt.xlabel('Frame Number')
    plt.ylabel('Average RGB Value')
    plt.legend()

    plt.tight_layout()
    plt.show()


def compute_and_plot_bvp(rgb_values, fs):
    """
    计算并绘制BVP信号

    参数:
        rgb_values: RGB值字典
        fs: 采样率(Hz)
    """
    # 选择绿色通道作为BVP信号
    green_signal = np.array(rgb_values['G'])

    # 平滑信号
    lowcut = 1
    highcut = 4.0
    filtered_signal = butter_bandpass_filter(green_signal, lowcut, highcut, fs, order=5)

    # 计算心率
    heart_rate = calculate_heart_rate(filtered_signal, fs)
    logger.info(f"Estimated Heart Rate: {heart_rate:.1f} BPM")
    print(f"----HR:{heart_rate}----")
    # 绘制BVP信号
    frames = range(len(green_signal))

    plt.figure(figsize=(12, 8))

    # 绘制原始绿色信号
    plt.subplot(2, 1, 1)
    plt.plot(frames, green_signal, label='Raw Green Signal', color='green')
    plt.title('Green Channel Signal Over Time')
    plt.xlabel('Frame Number')
    plt.ylabel('Average Green Value')
    plt.legend()

    # 绘制过滤后的BVP信号
    plt.subplot(2, 1, 2)
    plt.plot(frames, filtered_signal, label='Filtered BVP Signal', color='blue')
    plt.title('Filtered BVP Signal Over Time')
    plt.xlabel('Frame Number')
    plt.ylabel('Filtered Green Value')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()



