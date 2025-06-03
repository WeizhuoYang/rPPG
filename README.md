# 远程光体积脉搏波（rPPG）与实时血压估算

## 概述
该项目通过摄像头捕获面部视频流，提取前额和脸颊的RGB值来计算心率，并尝试使用无监督学习方法估算血压。具体功能包括：
- 实时检测人脸并提取感兴趣区域（ROI）。
- 计算心率并通过滑动窗口减少延迟。
- 使用KMeans聚类方法进行简单的无监督血压估算。
- 在视频流中实时显示心率和估算的血压。

## 依赖库
- OpenCV (`opencv-python-headless`)
- NumPy (`numpy`)
- SciPy (`scipy`)
- Scikit-Learn (`scikit-learn`)
- Matplotlib (`matplotlib`)

## 安装依赖库
在终端或命令提示符中运行以下命令来安装所需的Python库：
```bash
pip install opencv-python-headless numpy scipy scikit-learn matplotlib
```

## 下载Haar级联分类器
确保你有一个名为 `haarcascade_frontalface_default.xml` 的Haar级联分类器文件。你可以从OpenCV的GitHub仓库下载：
[Haar Cascade Classifier](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml)

将下载的文件放在项目的根目录下。

## 运行代码
在终端或命令提示符中导航到项目目录并运行以下命令：
```bash
python main.py
```

## 功能说明
1. **人脸检测**: 使用Haar级联分类器检测视频流中的最大人脸。
2. **ROI提取**: 提取前额和脸颊的ROI区域，并计算这些区域的平均RGB值。
3. **心率计算**: 使用5秒的滑动窗口和带通滤波器计算心率。
4. **血压估算**: 使用KMeans聚类方法对rPPG信号进行聚类，从而估算收缩压和舒张压。
5. **实时显示**: 在视频流中叠加心率和估算的血压信息。

## 示例输出
![Example Output](example_output.png)

## 注意事项
- 该方法只是一个基础演示，实际应用中需要更复杂的模型和更多的数据来提高准确性。
- 血压估算的结果仅供参考，不能用于医疗诊断。

## 日志记录
程序会记录详细的日志信息，帮助调试和监控运行状态。日志级别设置为INFO，可以通过修改代码中的日志配置来调整。

## 贡献
欢迎提交问题和建议！如果你有任何改进意见，请创建一个issue或pull request。

## 目录结构
```
remote_ppg_blood_pressure/
├── README.md
├── haarcascade_frontalface_default.xml
└── main.py
```

---

希望这个README文件适合在GitHub上使用。如果有任何进一步的需求或问题，请随时告知！
