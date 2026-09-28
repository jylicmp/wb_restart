# 第 0 轮部分结果

部分输出提供当前已完成 K 点的原始加权贡献，不按完成比例重新归一化。
权重覆盖率和点数完成率都不是积分误差估计。只有所有点有效时，快照才标记
`complete=true`；这个标记不代表后续自适应计算完成。

## 计算中输出

续算命令行及生产模板默认每 900 秒更新一次，有新增结果才写入。新计算的
第一个结果落盘后立即输出；恢复已有结果后也立即输出；第 0 轮结束时输出完整快照。
串行计算在点完成后检查时间，Ray 在收集/等待循环中检查，因此单个串行点耗时
超过 15 分钟时，更新时间也会延后。全部历史代次保留，不复制逐点结果。

```python
wb.run(system, grid, calculators,
       restart=True, restart_recover=True,
       file_Klist_path='/independent/checkpoint',
       fout_name='/independent/results/CrSe',
       partial_save_interval=900,
       partial_output_dir='/independent/results/partial')
```

Python API 默认 `partial_save_interval=0`，保持旧调用行为。启用后同时启用逐点检查点。
命令行用 `--partial-interval 0` 关闭，`--partial-output` 指定目录，默认是
`<output>.partial`。首版仅输出全局第 0 轮，支持 `EnergyResult` 计算器，包括当前 IMD/QMD。

## 只读导出已有结果

```bash
python /path/to/repo/scripts/restart_run.py \
  --config /path/to/repo/examples/crse_config.py \
  --checkpoint /independent/checkpoint --recover --iteration 0 \
  --export-partial --partial-output /independent/export
```

导出不会补算、启动 Ray、隔离损坏文件或修改输入检查点。缺失、损坏及读取过程中
变化的文件从汇总中排除并单独计数；配置冲突报错。导出默认标记动态快照。
零个有效结果只生成进度元数据，不伪造零张量。完整模型加载及网格重建仍应在
计算作业内执行；验收使用独立结果副本。原任务目录继续只读。

## 文件与读取

`latest.json` 原子指向一个完整的 `snapshot-*` 目录。目录包含：

- `metadata.json`：配置、完成点数、有效权重、缺失/损坏/变化数量、计算器映射和数据文件 SHA-256。
- `tensors.npz`：`completed_indices` 及每个计算器的能量轴、`data`、`dataSmooth`，不需要 pickle。
- `c0000.dat` 等：能量列及原始/平滑张量分量，保留 17 位小数格式；复数拆为实部和虚部。

计算器名称到 `c0000` 等前缀的对应关系在元数据里。张量分量按 NumPy C 顺序展平，
例如秩 2 的九个分量依次为 xx、xy、xz、yx、yy、yz、zx、zy、zz。

```python
import json
from pathlib import Path
import numpy as np

root = Path('/independent/results/partial')
latest = json.loads((root / 'latest.json').read_text())
generation = root / latest['snapshot']
metadata = json.loads((generation / 'metadata.json').read_text())
with np.load(generation / 'tensors.npz', allow_pickle=False) as data:
    prefix = metadata['calculators']['Qorb_IMD']['prefix']
    energy = data[prefix + '_energy_0']
    imd = data[prefix + '_data']
    imd_smooth = data[prefix + '_dataSmooth']
```

零有效点时没有 `tensors.npz`，应先检查 `completed_points`。读取时先读一次
`latest.json`，再读取该代次的文件；不要根据临时目录的修改时间挑选最新结果。
定时输出失败只警告并保留旧清单，下一周期重试；只读导出失败返回非零退出码。
重启遗留的未发布目录不自动删除。快照不是续算依据，丢失快照不影响逐点检查点。

输出目录有单写入者锁并绑定配置指纹，不能在同一目录混用模型。设置 `WB_SOURCE`
时，其父目录视为受保护的原任务树；输出也不能位于输入检查点内部或其祖先目录。
部分输出模块不纳入科学配置指纹，旧版续算检查点无需迁移。
