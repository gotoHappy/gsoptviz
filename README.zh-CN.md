# gsoptviz

同时查看优化过程中的 **3D 高斯和相机位姿变化**，支持浏览器交互播放、
离线 HTML 动画和 MP4 导出。查看器不依赖训练框架、PyTorch 或 CUDA。

完整输入协议、坐标约定和接口说明见 [English README](README.md)。

## 快速开始

需要 Python 3.10 及以上版本和支持 WebGL 的浏览器，在仓库目录运行：

```bash
python -m pip install -e .
python examples/make_demo.py
gsoptviz serve demo_sequence/manifest.json --port 8080
```

浏览器打开 <http://127.0.0.1:8080>。示例生成六帧合成圆环和相机轨迹，
包含高斯数量变化，不使用数据集或训练检查点。示例运动仅演示功能，不代表优化结果。

- 绿色：固定参考相机；橙色：当前相机。
- 蓝色：截至当前快照的相机轨迹；粉色：相机到参考位置的误差连线。
- 支持播放、暂停、逐帧、拖动时间轴、选择相机、切换观察视角。
- 高斯显示椭球形状、透明度和 SH0/DC 颜色，不渲染高阶 SH 的方向相关颜色。
- 默认显示全部高斯，只有指定 `--max-gaussians` 才抽样。

远程使用时，在本地电脑转发服务器端口：

```bash
ssh -N -L 8080:127.0.0.1:8080 user@host
```

## 导出

```bash
gsoptviz inspect demo_sequence/manifest.json
gsoptviz export demo_sequence/manifest.json --output artifacts/demo.html --fps 3
```

HTML 自带资源，打开后默认暂停，可用底部时间轴播放和跳转。MP4 需额外安装：

```bash
python -m pip install -e ".[video]"
python -m playwright install chromium
gsoptviz video demo_sequence/manifest.json --output artifacts/demo.mp4
```

输出文件必须尚不存在。长序列优先使用在线查看器或 MP4；离线 HTML 会预加载全部快照，
内存随总高斯数量增长。

## 接入训练

使用 `gsoptviz.Recorder` 传入固定参考相机、内参、图像尺寸和相机 ID，
再在保存步调用 `record(step, gaussian_arrays, current_c2w)`。
高斯和相机应来自同一时刻，Torch 张量先转换为 NumPy。
记录器要求新建输出目录，不覆盖已有记录。

高斯支持标准 3DGS PLY 或 NPZ；相机为 `(C,4,4)` 的 NPY 数组。
相机 ID、数量和顺序在序列中保持一致，高斯数量可以变化。
输入支持 OpenCV/OpenGL 和 c2w/w2c，记录器接口使用 OpenCV c2w。

默认所有数据共用一个世界坐标系，也可指定一次固定 `model_to_gt` 相似变换。
工具不做逐帧拟合对齐；位置误差单位为参考世界单位，旋转误差单位为度。
保存步之间不插值，播放速度表示每秒快照数。

基于 [Viser](https://github.com/nerfstudio-project/viser) 的浏览器渲染与场景序列化实现。

采用 [MIT 许可证](LICENSE)。
