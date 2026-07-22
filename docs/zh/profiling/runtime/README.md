# runtime

运行时集成层：处理“训练代码如何接入 profiling”。

## 你什么时候会改这里

- 调整框架无关接入逻辑（frameworkless API）。
- 增加/修改采集后端（capture backend）。
- 修改运行时配置结构（`NsysLaunchConfig` 等）。

## 关键文件

- `frameworkless.py`: 最常用运行时辅助函数（训练脚本可直接调用）。
- `config.py`: 运行时配置定义。
- `backends.py`: 采集后端抽象。
- `capture_controller.py`: 采集生命周期控制。
- `ProfileManager.py`: 运行时管理器。
- `template_utils.py`: 模板路径辅助。

## 实战建议

先尽量在 `frameworkless.py` 层接入，避免把 profiling 逻辑散落到训练代码多个位置。  
