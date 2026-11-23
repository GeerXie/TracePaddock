# TracePaddock

Offline analysis of recorded in-vehicle network data for ADAS and chassis control development.

面向 ADAS 与底盘控制开发的车载总线数据离线分析工具。

[English](#english) · [简体中文](#简体中文)

## English

TracePaddock opens MDF and MF4 recordings. Use it to compare signals, inspect CAN signal definitions from a DBC, and check how recorded states and outputs change over time.

The project grew out of practical work in chassis controls and automated driving. It draws on workflows common to several offline signal inspection and simulation tools used in those fields. Geer Xie built it with assistance from AI agents to make recurring tasks easier, such as reusing a signal layout, isolating an event in a long recording, and checking a response against the trace.

- Plot analog and digital signals, switch to XY view, and save display settings in a PLT.
- Compute derived signals, apply value mappings, and show named states as bands.
- Define event and response rules in the Analysis panel, or use Python scripts and plug-ins. Jump from a result to its location in the trace and export an event report to Excel.
- Load a DBC to reference message and signal definitions and import value tables for recorded channels with matching names.

### Install and start

**Windows portable build.** Extract `TracePaddock-0.5.2-win64.zip` from the [GitHub release](https://github.com/GeerXie/TracePaddock/releases/latest) and start `TracePaddock.exe` from the extracted folder.

**From source.** Use Python 3.10–3.12. Python 3.11 is recommended. Run these commands in the repository directory, preferably in a virtual environment.

```text
python -m pip install .
python -m tracepaddock
```

### Try the included measurements

Open **Help → Quick warm-up**. It walks through trace analysis, signal processing and state bands, and timing rules using three included synthetic recordings. The last exercise includes a DBC and a Python rule to try in the Analysis panel.

To start with your own recording, use **File → Open recording and view…** or drag the recording into the window. Supply a PLT to reuse an existing signal selection and layout. Press `E` to choose signals, and use **Analysis** to set up rules.

### License, citation, and feedback

Copyright © 2026 Geer Xie for original contributions in which he holds rights. The [project license](LICENSE) permits personal learning, noncommercial research, and internal training and evaluation among colleagues in a non-production setting without client deliverables. Adaptations can be shared under its terms with attribution, a copy of the license, and a clear record of the changes. Paid sale and other commercial uses require prior written permission. Third-party components have [their own terms](THIRD_PARTY_NOTICES.md).

If you reference TracePaddock in research, use the [citation file](CITATION.cff). Use this repository’s Issues tab for questions and reproducible reports. Include the TracePaddock version, your operating system, and the steps to reproduce the issue. Use the bundled synthetic data or a minimal example you can share publicly.

## 简体中文

TracePaddock 支持打开 MDF 和 MF4 录制，用于对照车辆信号、查阅 DBC 中的 CAN 定义，并检查状态和输出随时间的变化。

这个项目源于底盘控制和自动驾驶的实际开发工作。Geer Xie 借助 AI 智能体开发，参考了这些领域中多种离线信号分析工具和仿真工具的常用流程。项目针对日常复盘里的具体问题，例如复用信号布局、从长录制中定位事件，以及回到波形上核对输出响应。

- 绘制模拟量、数字量和 XY 曲线，将信号选择和显示配置保存为 PLT。
- 计算派生信号，使用数值映射把状态码显示为文字或状态带。
- 在 **分析** 面板中用表单、Python 脚本或插件配置触发和响应规则，从结果定位到波形，并导出 Excel 事件报告。
- 从 DBC 查看报文与信号定义，为录制中名称匹配的通道导入枚举值表。

### 安装和启动

**Windows 便携版。** 从 [GitHub 发布页](https://github.com/GeerXie/TracePaddock/releases/latest) 下载并解压 `TracePaddock-0.5.2-win64.zip`，在解压目录中运行 `TracePaddock.exe`。

**从源码运行。** 支持 Python 3.10–3.12，推荐 3.11。可先创建虚拟环境，然后在仓库目录中执行。

```text
python -m pip install .
python -m tracepaddock
```

### 用随附录制上手

打开 **帮助 → 快速上手**（Help → Quick warm-up）。三份合成录制分别练习信号分析、信号处理与状态带、时序规则。最后一项提供用于练习的 DBC 和可在分析面板预览的 Python 规则。

查看自己的录制时，用 **文件 → 打开录制和视图…** 或直接把文件拖进窗口。也可选用一份 PLT 来复用原有信号列表和布局。按 `E` 选择信号，通过 **分析** 菜单配置规则。

### 许可、引用和反馈

© 2026 Geer Xie，保留其对原创贡献依法享有的权利。许可覆盖个人学习、非商业研究，以及同事之间的非生产性内部培训与评估，内部用途不得涉及客户交付。可以按许可分享修改版本，须保留署名和许可文本，并明确标注改动。有偿销售和其他商业使用须事先取得书面许可。完整条款见 [LICENSE](LICENSE)，第三方组件适用[各自许可](THIRD_PARTY_NOTICES.md)。

研究中如需引用，请使用 [CITATION.cff](CITATION.cff)。问题和复现报告可发到本仓库的 Issues。请附 TracePaddock 版本、操作系统和复现步骤，并使用随附的合成数据或可公开共享的最小示例。
