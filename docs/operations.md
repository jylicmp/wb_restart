# 开发、验收与生产切换

## 隔离原则

原任务目录及其所有文件是只读数据源。环境修改仅部署到 `wberri_v1.8_restart`。
测试、日志、抽样结果和新检查点必须位于独立工作目录。不要对原临时目录调用
`wb.run()`：即使只是恢复，新版也需要写锁文件和检查点。

服务器原环境先通过 `conda create --copy --clone ... --prefix ...` 复制；
已逐文件核验 94 个包文件一致、无共享 inode。基线校验值在 `provenance/baseline.json`，
依赖版本在 `provenance/packages.json`。Git 里不保存模型、结果或集群私有配置。

## 验收

资源由 `sbatch` 参数指定，模板不固化站点信息。小模型作业使用单节点、4 CPU、
16 GiB、30 分钟；真实 CrSe 抽查使用单节点、4 CPU、256 GiB、最多两小时。
累计实际运行及待运行作业的上限不能超过批准的 4 节点小时。
可用 QOS 为 `qos30m`、`qos1h`、`qos6h`，同时明确指定相应作业时限。

通过环境变量指定 `WB_REPO`、`WB_ENV`、`WB_ORIGINAL_ENV`、`WB_WORK`；
CrSe 验收另需 `WB_TB_FILE` 和只读的 `WB_SOURCE`（原临时目录）。
`validate_crse.py` 复制权重及最多八个稳定结果文件，完整重建原网格并逐项比较权重，
通过真实 Ray worker 重算抽样点，比较原始和平滑后的 IMD/QMD 张量。
复制前后检查文件大小、时间、inode 和内容校验值；报告明确标记动态快照。
原程序仍会继续生成文件，报告中的文件数量不是冻结时刻的生产检查点。

最终验收同时检查 Slurm 终态、退出码、日志以及数值报告，不能只看作业离开队列。
失败作业不直接重投：先记录原因、修正一项，再提交到新的独立目录。

## 生产切换（本次不执行）

1. 由任务所有者安排原作业停止写入。原进程没有新版锁协议，不能依赖新版锁阻止它写入。
2. 保留原目录不变，将原 `_tmp_wb` 完整复制到新的、空的续算目录；禁止硬链接和指回原目录的符号链接。
3. 对源与副本逐文件核验名称、大小及 SHA-256，保存复制时间、输入脚本、模型和源代码校验值。
4. 使用新环境，在独立作业内运行 `restart_run.py --inspect --recover`。若配置、编号映射或抽查有差异，停止切换。
5. 设置 `WB_CHECKPOINT` 为副本、`WB_OUTPUT_DIR` 为新的结果目录，使用 `examples/resume.sbatch`。
   此模板拒绝把检查点或输出放在原任务树内。首次补算可选择 `recompute`，损坏副本将被改名保留。
6. 确认复用/缺失/损坏数量和 worker 版本，再检查第 0 轮输出及后续细化。入口默认续算到全局迭代 20。

模板默认单节点，由分配内的本地 Ray 管理资源。多节点生产资源尚未在本次任务中申请；
如要多节点运行，先按已有站点 Ray 启动方式建立该次分配专用集群，再运行入口。
小规模补算不必沿用原作业的 15 节点资源配置。

## 回滚

先停止使用新环境的作业，再回滚，避免 worker 混用版本。原环境无需回滚。

```bash
git worktree add --detach /path/to/rollback baseline-v1.8-mqm
python /path/to/current/repo/scripts/deploy.py \
  --checkout /path/to/rollback --environment /path/to/wberri_v1.8_restart
```

部署工具仅删除上次部署清单列出的、回滚版本不再包含的包文件。
新版检查点不能交给未修改的旧版直接读取。保留生产切换前的原格式备份；
不能以原版 `restart=False` 启动包含已有结果的目录，它会清空临时目录。

每次部署的实际文件哈希和 Git 提交写入包内 `.wb_restart_deployment.json`。
正式验收要求干净提交；`--allow-dirty` 仅用于开发测试，并被明确标记。
