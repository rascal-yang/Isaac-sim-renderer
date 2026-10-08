本仓库迁移用户工作区中已经验证的渲染实验实现；原文件哈希见 docs/source_provenance.json。它不是 NVIDIA 官方 CloudXR 实现，也不重新授权第三方模型、资产和源码。

官方触觉算法来自 IsaacLab 的 isaaclab_teleop/haptic_feedback.py；完整 BSD-3-Clause 许可证见 integrations/g1/vendor/LICENSE-IsaacLab。兼容适配不改上游源文件。

完整交付资源的出处与许可证见 resources/sonic-grasp-augmentation/NOTICE.md、licenses/ 和 source_manifest.json；SONIC/模型/真实场景资产须遵守其原许可。随包的 ONNX Runtime 许可证位于 resources/onnxruntime/onnxruntime/LICENSE 和 ThirdPartyNotices.txt。Git 本身不包含这些大资源。

交付没有 API 密钥。固定 token 回放不访问 GPT。
