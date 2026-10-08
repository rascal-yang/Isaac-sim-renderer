# 固定默认配置与接口

当前入口维护原实验的已验证配置；G1 场景适配在 `integrations/g1`，入口/协调/交付在仓库根。第三方模型和资产仅作为资源依赖，不进入 Git。没有修改旧实验目录。

| 参数 | 默认值 |
|---|---|
| 物理 / SONIC / 视觉 / 官方触觉 | 200 / 50 / 50 / 50 Hz |
| 控制 decimation | 4 次 5 ms 物理步 / 20 ms 控制步 |
| PhysX | CPU, MBP, TGS, 8 threads; GPU dynamics false |
| G1 求解迭代 | position 8 / velocity 4 |
| 双手求解迭代 | position 32 / velocity 8 |
| 重力 | 全部恢复；质量、摩擦、碰撞、PD 与原方案一致 |
| ONNX release decoder | CPU execution provider, intra-op 2 / inter-op 1 |
| RT2 | RealTimePathTracing，rt2 enabled，原生 640×480 DLAA，AA op 4 |
| 渲染更新 | 每 50 Hz 状态 1 次同步 render；初始 4 次预热 |
| Ray bounce | max 4，specular/transmission 6；virtual motion true |
| frame generation / async / multi-GPU | false / false / false；GPU 0 |
| waitIdle / Fabric / USD 逐步写入 | true / true / false |
| 材质与光照 | 原资产，translucency/reflections true，exposureKey 0.6 |
| 相机 | torso_link; local [0.095,0,0.45] m; 下俯 35° |
| 镜头 | 18 mm；aperture 36×27；clipping 0.015–40 m |
| MP4 | H264, CRF 18, fast, 2 threads, yuv420p, 50 fps |
| 编码缓冲 / 状态在途缓冲 | 16 / 8 帧有界 |
| CPU 分配 | 按 sysfs 物理核心分组；超线程兄弟不跨进程 |
| 其他线程 | 重叠模式 OMP/MKL 2、OpenBLAS 1、Kit tasking 8 |

运行时参数读回保存到 `renderer_settings*.json`、`physx_boundary_config.json`、`all_gravity_audit.json`、`xr_runtime_config.json` 等；上表不是替代读回。同步模式沿用原不绑核对照。加速结果属于结构与线程调度的组合，不能单独归因于一个参数。

## 资源契约

`resources/sonic-grasp-augmentation` 是带版本和哈希的 G1 场景/模型依赖；其 `prepare` 物化路径模板到新 runtime。模板与运行时绝对路径用于解析 USD 纹理，交付源码不依赖作者工作区。`resources/onnxruntime` 是原 CPython 3.11 Linux 的 1.30.0 CPU 运行库。`demo_candidate` 为已有 token，不在线调用 GPT、planner 或 encoder；decoder 仍按 50 Hz 执行。

## 状态与帧

N 个控制步产生 4N+1 行物理公开状态及原始接触、N+1 行图像源状态和相机矩阵。索引 0 是初始状态；视频第 i 帧（从 0 计）恢复第 i+1 行，与执行后 (i+1)/50 秒对应。

结构化 NumPy dtype 记录 robot、brainco_left、brainco_right 的 root_state_w(13)、joint_pos/vel、三类 joint targets、全部 body poses/velocities；object/table 的根和连杆状态；额外活动刚体的 transform_xyzw/velocity。schema 保存全部刚体覆盖与静态 Xform。禁止有未覆盖 deformable/rigid collection。

32×3×3 世界坐标接触力矩阵与 32×3 净接触力保存 200 Hz 牛顿值及过滤映射；64 桌面/房间监测通道持续检测；官方两手 10 指触觉反馈保存 50 Hz。逻辑接触覆盖未删减。渲染进程不重新计算源物理任务/触觉。

相机记录 4×4 float64 世界矩阵。渲染时恢复根、关节、目标、活动刚体与相机，调用 sim.render 一次更新 Fabric。每帧 FK readback 与已完成渲染的 camera_params 分别检查；PhysicsContext._step 被替换为抛错，禁止额外物理推进。kinematic table 只写 pose，避免给运动学刚体写 velocity 的上游异常。

公开状态不是 PhysX 私有接触/暖启动缓存；用于重现画面，不用于证明可以恢复动力学继续积分。

## 流水线与计时

UNIX socket 发送 8-byte network-order int64 帧号，接收者渲染完成回相同 ACK；-1 为结束。源队列最多 8 帧；关闭先读尽所有 ACK 再关连接，避免末尾丢帧。内存映射在发布索引前写完对应行；ACK 是数据可消费的同步边界。持久化 flush 在收尾。

两进程仍分别启动 Isaac rendering.kit；CPU 物理端不建立轨迹 render product，但仍占 GPU 上下文。这不是无 GPU 机器执行验证。没有多轨迹并发功能。

active pipeline = rollout_start_perf 至 max(artifacts_finished_perf, render_finish_perf)，包含执行、排队及视频收尾；cold = coordinator 启动至两进程退出，包含 Kit/模型初始化。`render_s` 包括 Kit/Fabric 与等待，不是纯 GPU 内核时间。各进程阶段重叠，不能直接相加当作墙钟总时间。

RT 采样像素不保证逐位相同；物理文件与相机矩阵可以严格比较，图像用同帧 SSIM/差异和错帧检测评估。训练数据的全面视觉认证仍需单独完成。
