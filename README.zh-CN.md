# WannierBerri 1.8 续算版

[English](README.md) | **简体中文**

本项目在现有 WannierBerri **1.8.0 定制版本**上增加中断续算、部分结果导出和有界 Ray 结果收集。
原有 MQM 计算器、公式及矩阵配置保持不变；完整基线见 `baseline-v1.8-mqm`。
上游代码按 [GPL](LICENSE) 分发。

## 功能

- 每轮分发计算前提交 K 点清单和权重；支持首轮及自适应迭代中断续算。
- 结果原子落盘、身份和配置校验、损坏文件隔离、目录迁移及单写入者锁。
- 显式恢复缺少 `K_list.pickle` 的旧版第 0 轮，复用有效结果，只补算缺失点。
- 第 0 轮计算期间定期发布完整性可验证的部分结果快照。
- Ray 使用引用到 K 点的映射收集结果，待处理任务和反序列化批次均有上限。
- 增量自适应细化只检查新 K 点的对称等价关系，避免在大型旧网格上进行无效的二次扫描。

## Python 接口

```python
result = wb.run(
    system, grid, calculators,
    restart=True,
    restart_recover=True,        # 仅缺少清单的旧版第 0 轮需要
    restart_on_corrupt="error",  # 或 "recompute"：隔离损坏文件后补算
    file_Klist_path="/path/to/independent/checkpoint",
    allow_restart=True,
    dump_results=True,
    partial_save_interval=900,
    partial_output_dir="/path/to/partial-results",
    adpt_num_iter=20,
)
```

`adpt_num_iter` 保留原语义：从选中的全局迭代起，额外执行这么多次细化。
例如从迭代 5 接续至迭代 20，应设置 15。命令行入口自动换算：

```bash
export WB_TB_FILE=/path/to/CrSe_SOC_tb.dat
python /path/to/repo/scripts/restart_run.py \
  --config /path/to/repo/examples/crse_config.py \
  --checkpoint /path/to/independent/checkpoint --recover --inspect

python /path/to/repo/scripts/restart_run.py \
  --config /path/to/repo/examples/crse_config.py \
  --checkpoint /path/to/independent/checkpoint --recover \
  --on-corrupt recompute --until-iteration 20 --output /path/to/results/CrSe \
  --partial-interval 900 --partial-output /path/to/partial-results
```

检查命令只读，但完整模型加载和网格重建仍需通过计算作业执行。`--inspect`
报告有效、缺失和损坏编号；配置或结果结构不一致会报错，不会当作损坏而补算。

## 安装、测试与切换

先克隆原环境为 `wberri_v1.8_restart`，校验源码一致及文件独立，再从干净 Git 提交部署：

```bash
python scripts/deploy.py --environment /path/to/wberri_v1.8_restart
```

部署会核对文件内容并记录 Git 提交，不使用 editable install。脚本拒绝部署到其他名称的环境。
运行时应离开仓库根目录，避免当前目录中的源码遮蔽环境安装。

- [第 0 轮部分输出与只读导出](docs/partial-output.md)
- [验收报告：首轮专项、真实 CrSe 与 Ray](docs/validation.md)
- [检查点格式与兼容边界](docs/checkpoint-format.md)
- [生产切换、测试和回滚](docs/operations.md)
- 小模型测试：`python -m unittest discover -v -s /path/to/repo/tests`
- 原版/新版/真实 Ray 验收模板：`examples/validate_small.sbatch`
- 真实网格和抽样重算模板：`examples/validate_crse.sbatch`

持久续算当前支持 `System_R`（包括 `System_tb`）；无检查点的计算保持原有系统类型支持。
旧 pickle 不包含坐标和完整配置指纹，重建权重一致仍不足以独立证明其来源。
恢复前需要核对原始参数，并重算少量点验证；只加载可信的 pickle。
