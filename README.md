# nano2 — 在 64 KB Flash 的 STM32F103C8T6 上跑 int4 量化小语言模型

> **改进自 [egger435/egg-nano](https://github.com/egger435/egg-nano)**：原作者在 STM32F103C8T6 上用 int8 量化跑了一个 4.2 万参数的 byte-level GPT（推理核 ~275 行 C）。
> 本项目保留"极限端侧"的前提（**64 KB Flash / 20 KB SRAM、无操作系统、无外部算力**），做了四件事：
> 1. 量化从 int8 换成 **int4 + QAT（STE 伪量化）**
> 2. 注意力从 MHA 换成 **GQA**（4 个 Q 头共享 2 组 KV）
> 3. 注意力再叠一层 **滑窗稀疏 + 环形 KV**（Longformer 式局部窗口 + 全局首 token）
> 4. 自建语料、从头训练，并补上**中文显示**（12×12 点阵字库）

---

## 结果一览

| 项 | 值 |
|---|---|
| 模型 | 3 层 decoder，d_model 48，4 Q 头 ×12，GQA 2 组 KV，FFN 128，vocab 256（byte-level），窗口 72 |
| 参数 | **76,128**（int4 权重 69,888 + pos_emb 4,608 int8 + f32 伴随 1,632） |
| Flash / RAM | **63,060 B / 65,536（96.2%）**，**17,496 B / 20,480（85.4%）** |
| 训练 | Windows CPU（i5-13500H，16 线程，PyTorch 2.15），945 条自建语料，500 epoch ≈ 35 min |
| 验证 | 教师强制字节准确率 100%；**宿主直跑固件 C 代码 44/44**；真机串口问答通过 |
| 生成速度 | 短答 ~0.9 s，长答 ~5 s（瓶颈是软件 IIC 逐字符刷屏） |

---

## 快速开始

```bash
# 1) 生成语料（公开版本用占位人名）
python src/corpus_v3.py --public

# 2) 训练（CPU 即可，约 35 分钟）
python src/train.py --epochs 500 --bs 64 --lr 3e-3

# 3) 导出 int4 权重 C 数组
python src/export_c.py            # 生成 fw/src/n2_weights.{c,h}

# 4) 构建固件（需要 arm-none-eabi-gcc）
cd fw && python build_n2.py --clean

# 5) 烧录（ST-Link + OpenOCD；Windows）
powershell -ExecutionPolicy Bypass -File tools/flash_n2.ps1

# 6) 真机验收（CH340 串口）
python src/test_board_full.py COM12
```

> 提示：如果 SWD 连不上（现象是能读芯片 ID 但内存访问全失败），先确认固件里有没有开 `DBGMCU_CR` 的 `DBG_SLEEP/STOP/STANDBY`——没有的话 MCU 一进 `__WFI()` 休眠，AHB-AP 就不可访问了。详见 [docs/报告.md](docs/报告.md) 第 8 节。

---

## 目录

```
.
├─ docs/报告.md            完整设计/训练/部署报告（含 Flash/SRAM 预算、踩坑记录）
├─ docs/loss_curve.png     训练曲线（交叉熵 + 贪心命中率双轴）
├─ src/
│   ├─ corpus_v3.py        语料生成器（人物关系 / 闲聊 / 数学 / 边界拒绝）
│   ├─ model.py            GQA + 滑窗 + int4 QAT 的 PyTorch 实现
│   ├─ train.py            训练 + 评估（教师强制 & 贪心）
│   ├─ export_c.py         ckpt → int4 打包的 C 数组（+ int16 bias）
│   ├─ sim_mcu.py          numpy 复刻固件整数路径（烧录前一致性闸门）
│   ├─ build_host.py       把固件 C 代码编成 PC 程序
│   ├─ acceptance_host.py  在 PC 上跑固件 C 代码的用例集
│   ├─ test_board_full.py  真机串口验收
│   ├─ make_ui_preview.py  把 OLED 排版在 PC 上渲染成 PNG
│   ├─ plot_loss.py        画 loss 曲线
│   └─ tools/make_cjk_font.py  12×12 中文字库生成器（Pillow + 等线）
├─ fw/                     固件工程（C99，STM32F1 标准外设库）
│   ├─ src/n2_model.c      int4 推理核（GQA + 环形滑窗 KV + LUT GELU）
│   ├─ src/n2_ui.c         OLED 排版（ASCII 6×8 + 汉字 12×12 跨页绘制）
│   ├─ src/n2_font.c       自动生成的 79 字点阵字库
│   ├─ src/main.c          串口收问题 → 片上推理 → 串口 + OLED 输出
│   └─ build_n2.py         构建脚本（等价 EIDE Debug 目标）
└─ tools/                  烧录 / ISP / 串口小工具
```

`fw/src/n2_weights.{c,h}` 由 `export_c.py` 生成，仓库里不提交（约 240 KB 文本）。

---

## 硬件接线

| 外设 | 接线 |
|---|---|
| ST-Link V2（SWD + 供电） | `3.3V / SWDIO(PA13) / SWCLK(PA14) / GND` |
| SSD1306 OLED | `VDD→3.3V`、`GND→GND`、`SCL→PA6`、`SDA→PA5`（软件 IIC） |
| CH340（串口） | `TXD→PA10`、`RXD→PA9`、`GND→GND`，115200 8N1 |

---

## 许可与致谢

- 推理核的**原始设计**来自 [egger435/egg-nano](https://github.com/egger435/egg-nano)（byte-level GPT、int8 量化、OLED 排版思路），本项目在其基础上改了量化方案、注意力结构与语料，并重写了推理核与显示层。
- `fw/src/OLED.c` 的 6×8 字模与 SSD1306 驱动沿用原作者仓库中的 Mini-OLED 实现。
- 语料为合成数据，示例人名全部为占位符，与实际人员无关。
