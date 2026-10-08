# Isaac State Renderer

维护已实测的 G1 仿真加速管线：**CPU 物理及 SONIC → 200 Hz 完整公开状态与接触记录 → 有界队列 → GPU 恢复 50 Hz 状态并渲染 → MP4**。两个独立 Isaac 进程重叠执行，物理时序保持不变。

默认保持 200 Hz 物理、50 Hz SONIC、全部重力、原模型/碰撞/双手求解配置和 RT2 原生 640×480 DLAA。每个状态同步渲染一次，开始预热四次；材质与贴图来自原资产。

## 接收者快速开始

推荐拿完整 `isaac-state-renderer_20261009.tar.gz` 和同名 `.sha256` 文件。**Git 仓库只保存代码；Git clone 本身没有模型、场景、贴图和示例数据。完整交付包包含这些资源。** 本次仅创建本地 Git 仓库，没有发布远端。

已验证的平台：Linux x86_64、Python 3.11.15、Isaac Sim 5.1.0、IsaacLab source commit `37ddf626871758333d6ed89cf64ad702aef127d0`（安装包 metadata 为 0.54.2）、PyTorch 2.7.0+cu128、RTX 4090、32 个逻辑 CPU。版本见 `environment-observed.json`。原环境 OpenCV 4.13 使用非标准 NumPy 1.26 组合；安装清单选择兼容 NumPy 1.x 的 OpenCV 4.11（仅用于图像比较等，Isaac 渲染参数不变）。首次安装环境需要联网下载 Isaac/PyTorch；资产与模型已收集，运行无需 GPT API。

```bash
sha256sum -c isaac-state-renderer_20261009.tar.gz.sha256
tar -xzf isaac-state-renderer_20261009.tar.gz
cd isaac-state-renderer
# 全新环境：先安装系统 ffmpeg 和 Python 3.11
sudo apt-get install ffmpeg
SIM_PYTHON=python3.11 bash setup_environment.sh
source .venv/bin/activate
```

已有兼容 Isaac 环境可跳过安装，使用该环境的 `python`。不要额外安装独立 `usd-core`，它会与 Isaac 的 USD 冲突。`setup_environment.sh` 未在另一台空白机器上完整实测；本次验证是同机、新资源 runtime、仓库外启动。

```bash
python launch.py doctor
# runtime 与输出路径不要含小写 robot 子串：继承的上游 motion loader
# 会替换完整路径中的 robot→meta。路径选择后不要移动 runtime。
python launch.py prepare --runtime /tmp/isr-demo/runtime
python launch.py pipeline --runtime /tmp/isr-demo/runtime --output /tmp/isr-demo/overlap_01
```

最后一条命令使用交付包中的 **21.42 秒、1071 控制步**固定 token 示例，自动执行并验证全部视频帧、相机和状态恢复。交付资源 `resources/demo_video/head.mp4` 提供本次验证的视频参考。成功后查看：

- `/tmp/isr-demo/overlap_01/render/head.mp4`：640×480、50 fps、21.42 秒完整视频。
- `physics/lean_result.json`：任务是否成功，与速度分开判断。
- `verification.json`：物理/帧数、完整解码、逐帧状态与相机检查。
- `timing_summary.json`：直接区分 active_s、cold_process_s、物理、SONIC 和渲染各阶段。
- `process_report.json`：含进程启动与退出的冷启动用时与 CPU 分组。
- `physics/state_recording_timing.json` 与 `render/render_profile.json`：执行、恢复、渲染、队列和收尾用时。

输出目录必须全新，失败日志保留。第一次启动会编译/加载着色器和模型，冷启动与执行计时分别报告，不能把二者混用。文件编码、资源校验与验收也有开销。预留至少 15 GB 可用磁盘；每次完整运行仅原始 RGB 就约 1 GB。

## 重跑对照和离线渲染

```bash
# 同等完整数据保存的 CPU 物理 + 同步 RT2 渲染对照
python launch.py pipeline --runtime /tmp/isr-demo/runtime --scheme baseline --output /tmp/isr-demo/baseline_01
# 严格比较物理公开状态、接触、触觉、相机文件哈希
python launch.py verify --output /tmp/isr-demo/overlap_01 --reference /tmp/isr-demo/baseline_01
# 图像质量和相对错帧检查（先完成两组运行）
python compare_images.py --reference /tmp/isr-demo/baseline_01/baseline --candidate /tmp/isr-demo/overlap_01/render --output /tmp/isr-demo/image_comparison
# 已有状态只重渲染，不再次推进动力学
python launch.py render --runtime /tmp/isr-demo/runtime --source /tmp/isr-demo/overlap_01/physics --output /tmp/isr-demo/render_01
# 也可不跑物理，直接使用交付包内的公开状态样例
python launch.py render --runtime /tmp/isr-demo/runtime --source resources/demo_state --output /tmp/isr-demo/sample_render_01
```

任意工作目录均可用绝对路径调用 `launch.py`。资源默认根据入口文件位置查找；也可在子命令前加 `--resources /path/to/resources`。迁移仓库后建立新 runtime，因为 USD/配置文件在 prepare 阶段物化为当前机器路径。

自己的轨迹使用 `pipeline --candidate /path/to/candidate`；必须包含 `grasp_tokens.npz`、`initial_state.npz`、`deployment_parameters.json`、`plan.json` 与 `prepared_files.json`，并符合原 G1 token/关节顺序契约和 50 Hz 时间格。帧数从 token 长度推导，不固定为 1071。此仓库当前支持原 G1 + Revo2 场景；换机器人或场景需要实现资产构建适配器，不能只替换 tokens。

## 维护和交付

```bash
python -m unittest discover -s tests -v
python integrations/g1/test_stream_close.py
python pack.py --output /outside/repository/isaac-state-renderer_20261009.tar.gz
```

`resources/manifest.json` 是交付资源哈希清单；`prepare` 校验 G1 资产，物化后的依赖也再次检查完整性。资源依赖的许可证见 `resources/sonic-grasp-augmentation/licenses/`、`NOTICE.md` 和 `integrations/g1/vendor/LICENSE-IsaacLab`。大文件不进入 Git。

配置与数据契约见 [PIPELINE.md](docs/PIPELINE.md)，实际验收见 [REPRODUCTION.md](docs/REPRODUCTION.md)。原实验脚本未修改；迁移来源哈希见 `docs/source_provenance.json`。

Git 历史另附 `isaac-state-renderer_20261009.git.bundle`，可用 `git clone /path/to/isaac-state-renderer_20261009.git.bundle source-code` 获取代码仓库，再用 `--resources` 指向解压交付包的 resources。无需远端账户。
