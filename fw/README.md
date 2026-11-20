# 固件

```bash
python export_c.py          # 先在 ../src 下生成 fw/src/n2_weights.{c,h}
python build_n2.py --clean    # 需要 arm-none-eabi-gcc（见脚本顶部 TOOLCHAIN 变量）
```
