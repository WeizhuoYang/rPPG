import os
import cv2
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.signal import butter, filtfilt
import pywt
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
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

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

# FFT计算心率
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


def estimate_blood_pressure(rppg_signal):
    """
    简单的无监督方法估算血压

    参数:
        ppg_signal: rPPG信号

    返回:
        estimated_systolic: 估算的收缩压
        estimated_diastolic: 估算的舒张压
    """
    # 使用KMeans聚类来识别两个主要的心率变化点
    kmeans = KMeans(n_clusters=2, random_state=42).fit(np.array(rppg_signal).reshape(-1, 1))
    clusters = kmeans.labels_

    # 假设较高的集群是收缩压，较低的是舒张压
    systolic_cluster = np.max(clusters)
    diastolic_cluster = np.min(clusters)

    # 计算收缩压和舒张压
    estimated_systolic = np.mean(rppg_signal[clusters == systolic_cluster])
    estimated_diastolic = np.mean(rppg_signal[clusters == diastolic_cluster])

    return estimated_systolic, estimated_diastolic

def main():
    # 设置工作目录为脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # 直接指定输入和输出路径
    input_path = 0  # 使用默认摄像头

    # 创建RemotePPG对象
    rppg = RemotePPG(buffer_size=120, sampling_rate=30)

    # 初始化视频捕获
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        logger.error("无法打开摄像头")
        return

    # 获取视频参数
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 更新采样率
    rppg.fs = fps

    logger.info(f"视频信息: {frame_width}x{frame_height}, {fps} FPS, {total_frames} 帧")

    # 存储RGB值
    forehead_rgb_values = {'R': [], 'G': [], 'B': []}
    cheek_rgb_values = {'R': [], 'G': [], 'B': []}

    # 初始化计时器
    start_time = time.time()
    frame_idx = 0
    last_heart_rate = 0
    last_systolic = 0
    last_diastolic = 0
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

            # 存储每一帧的RGB值
            if mean_rgb_forehead is not None:
                forehead_rgb_values['R'].append(mean_rgb_forehead[2])  # 红色
                forehead_rgb_values['G'].append(mean_rgb_forehead[1])  # 绿色
                forehead_rgb_values['B'].append(mean_rgb_forehead[0])  # 蓝色

            if mean_rgb_cheek is not None:
                cheek_rgb_values['R'].append(mean_rgb_cheek[2])  # 红色
                cheek_rgb_values['G'].append(mean_rgb_cheek[1])  # 绿色
                cheek_rgb_values['B'].append(mean_rgb_cheek[0])  # 蓝色

            # 滑动窗口计算心率
            window_size = rppg.buffer_size
            if len(forehead_rgb_values['G']) >= window_size:
                green_signal = np.array(forehead_rgb_values['G'][-window_size:])
                filtered_signal = butter_bandpass_filter(green_signal, 1, 4.0, rppg.fs, order=5)
                # 小波去噪
                wavelet = 'sym8'
                level = 5
                coeffs = pywt.wavedec(filtered_signal, wavelet, level=level)
                medians = [np.median(np.abs(c)) for c in coeffs[1:]]
                threshold = np.median(medians) / 0.6745
                # threshold = 0.1
                # denoised_coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
                denoised_coeffs = [coeffs[0]] + [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
                wavelet_signal = pywt.waverec(denoised_coeffs, wavelet)
                last_heart_rate = calculate_heart_rate(wavelet_signal, rppg.fs)

                # 估算血压
                last_systolic, last_diastolic = estimate_blood_pressure(wavelet_signal)
                logger.info(f"Estimated Systolic BP: {last_systolic:.1f}, Diastolic BP: {last_diastolic:.1f}")

            # 在帧上显示心率
            cv2.putText(result['display'], f"Heart Rate: {last_heart_rate:.1f} BPM", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(result['display'], f"Systolic: {last_systolic:.2f}, Diastolic: {last_diastolic:.2f} ", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            '''
            # 显示RGB信息
            if mean_rgb_forehead is not None:
                cv2.putText(result['display'], f"Forehead RGB: {mean_rgb_forehead}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            if mean_rgb_cheek is not None:
                cv2.putText(result['display'], f"Cheek RGB: {mean_rgb_cheek}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            '''
            # 显示结果图像
            cv2.imshow('实时处理', result['display'])

        else:
            # 如果没有检测到人脸，直接显示原始帧
            cv2.imshow('实时处理', frame)

        # 按'q'退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

    logger.info(f"视频处理完成，共处理 {frame_idx} 帧")

    plot_rgb_changes(forehead_rgb_values, cheek_rgb_values)

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

if __name__ == '__main__':
    main()
