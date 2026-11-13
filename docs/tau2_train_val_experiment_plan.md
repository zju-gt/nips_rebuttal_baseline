# tau2-bench 70/30 train/val 分离实验计划

## 目标与比较口径

回应审稿人对训练与评测分离的要求：在**当前实验实际评测的 tau2-bench `base` split** 中，对 Airline、Retail、Telecom 分别固定划出 70% 训练任务和 30% 验证任务。AWM、ReasoningBank、Living-Harness 仅用训练任务生成各自的 harness / memory；训练完成后分别保存最终产物，再加载该产物，在未见过的验证任务上评测。GPT5.2-base 不训练，在同一验证集上直接评测。

主表采用每个验证任务一次运行的成功率（奖励为 1 的任务数 / 该领域验证任务数，单位 %）。四个方法使用相同的验证任务 ID、GPT-5.2 actor、用户模拟器、判分器、随机种子及运行配置；`Avg` 为三个领域成功率的**等权算术平均**，避免 Telecom 的任务数量主导总分。训练集上的成绩不进入主表。主表格式如下，数值在实验完成后填写：

| Method | Airline | Retail | Telecom | Avg |
| --- | ---: | ---: | ---: | ---: |
| GPT5.2-base | 待测 | 待测 | 待测 | 待测 |
| AWM | 待测 | 待测 | 待测 | 待测 |
| ReasoningBank | 待测 | 待测 | 待测 | 待测 |
| Living-Harness | 待测 | 待测 | 待测 | 待测 |

## 数据划分

数据源为当前 vendored tau2-bench 三个领域各自的 `data/tau2/domains/<domain>/tasks.json` 和 `split_tasks.json`，以现有 runner 使用的 `base` split 中**全部唯一 task ID** 为各领域划分母集。Airline 为 50 条，Retail 为 114 条，Telecom 为 114 条。尤其不能把 Telecom `tasks.json` 中的 2285 条全量任务混入本实验。原有 `split_tasks.json` 中的 `train` / `test` 不是本实验所需的 70/30 划分，不直接复用。

采用固定种子 `300`，每个领域独立划分。对 `base` 中每个 task ID 计算 `SHA256("300:<domain>:<task_id>")`，按摘要及 task ID 排序；前 `floor(0.7 × N + 0.5)` 个进入训练集，其余进入验证集。保存三个领域的 `base` ID 列表、train/val ID 列表、原始 `tasks.json` 与 `split_tasks.json` 的 SHA256 和划分算法版本；运行前锁定这些文件，所有方法读取同一份划分。检查 ID 唯一、训练/验证交集为空、并集等于对应的 `base` ID 集合。

| 领域 | 当前 `base` 任务数 | 训练 | 验证 |
| --- | ---: | ---: | ---: |
| Airline | 50 | 35 | 15 |
| Retail | 114 | 80 | 34 |
| Telecom | 114 | 80 | 34 |
| **合计** | **278** | **195** | **83** |

这些数量基于当前仓库的 `base` split；若 tau2-bench 数据版本变化，必须先重新生成并审查划分，再启动实验。主表仅使用上述 83 条验证任务（每个方法各评测一次）；正式启动前仍需用小样本测算时间和 API 费用。

## 方法、训练和冻结

每个领域独立训练一个 harness，不跨领域共享状态。保留当前 `baseline_registry.py` 中各方法的核心设置；训练阶段 AWM 和 ReasoningBank 各运行 1 个 cycle，Living-Harness 运行 3 个 cycle。每个训练 task 的 reward / trajectory 可用于**所属方法、所属领域**的状态更新。训练阶段可使用方法内部的样本并发；方法间任务按顺序运行，以便控制 API 负载。训练作业完成后，导出最后一个 cycle 的**最终**状态，不能以中途快照充当最终 harness。

| 方法 | 训练产物，按领域分别保存 | 验证时使用方式 |
| --- | --- | --- |
| GPT5.2-base | 无训练产物 | 直接使用 `base_lm` prompt 与 GPT-5.2 |
| AWM | `workflow_memory.json` 及其实际推理依赖的 prompt / 配置 | 加载训练后的 workflow memory，保留检索，禁止写入 |
| ReasoningBank | `reasoning_bank.json` 及其实际推理依赖的 prompt / 配置 | 加载训练后的 reasoning bank，保留检索，禁止写入 |
| Living-Harness | `memories.json`、`state_graph.json`、训练时生成的 benchmark/domain prior 及配置；`meta_sop.md` 作为空占位文件保存以供审计 | 加载训练后的 memory、state graph 与 prior，保留检索，禁止写入 |

导出时以实际推理读取的字段为准：如果某方法还依赖其他持久化文件，也一并纳入该方法的产物清单。Living-Harness 的动态 prior / prompt 生成只允许看训练任务；验证时只能读取训练阶段保存的文本，不能拿验证任务重新挑选 representative tasks 或重新生成 prior。

**已确定的设置：本实验不更新 Meta-SOP。** 现有 Living-Harness tau2 runner 使用 `MODE=prior`，不会生成初始 Meta-SOP，训练循环也不会更新 `current_meta_sop_text`。因此 `meta_sop.md` 为空，仅用于记录当前实现；产物清单的 `meta_sop_empty` 字段标明这一点。Living-Harness 的训练产物是 memory、state graph 及训练集生成的 prior，不将空 Meta-SOP 描述为已学习的组件。

建议每个产物目录包含 `artifact_manifest.json`：方法、领域、split ID/hash、代码与数据版本、模型、种子、训练配置、产物文件及各自 SHA256、训练完成时间和训练样本数。保留完整训练日志、原始结果和 `api_usage.jsonl`，使成功率和 token/cost 都可追溯。

## 验证协议与防泄漏检查

在独立的验证进程中加载已冻结的 harness，每个验证 task 运行一次，即 `PASS_HAT_K=1`，固定种子 `300`；Living-Harness 验证阶段也只运行一轮，不再执行训练用的 3 个 cycle。对于 AWM、ReasoningBank 和 Living-Harness，需要新增明确的 **eval/frozen 模式**：保持加载与检索逻辑，但关闭 experience extraction、memory / graph / workflow / bank 更新及任何验证结果回写。仅设置 `EVOLVE_COMPONENTS=""` 不够，因为这也可能关闭推理所需的状态检索。验证任务之间共享同一份只读训练产物，不共享新的验证经验。

运行层面按每分钟 20 次 API 调用限速，验证样本的墙钟超时设置为 2400 秒，避免并发对话在等待限速配额时触及旧的 800 秒超时。模型、任务、最大交互步数和判分规则不变；运行摘要记录实际超时与限速配置。

验证开始前记录产物 hash，结束后再次核对；运行记录中写入加载的产物路径、hash 与 split hash。每个方法/领域的实际 task ID 集合必须与验证 split 精确相等且每个 ID 恰好出现一次。出现 infrastructure error 或缺失 reward 时，不将它们悄悄计作方法失败：先保留原始记录，再在相同 task ID、相同种子、相同冻结产物下补跑，主表只在验证样本完整且基础设施错误为 0 时发布。方法性失败（reward 非 1）仍计入分母。

所有方法的模型角色沿用当前 API runner：actor/evolution 为 `gpt-5.2-2025-12-11`，tau2 用户模拟器和自然语言判分器为 `gpt-5.1-2025-11-13`。主表写作 `GPT5.2-base` 是 actor 方法名；实验记录需保留这两个模型的具体版本及接口配置。各方法的训练成本、验证成本、token 消耗和耗时另列附表，不混入成功率主表。

## 文件布局与实施步骤

```text
code/results/<train-val-run-id>/
  split/
    split_manifest.json
    airline.json
    retail.json
    telecom.json
  train/
    awm/<domain>/...
    reasoningbank/<domain>/...
    living_harness/<domain>/...
  harnesses/
    awm/<domain>/artifact_manifest.json + 产物文件
    reasoningbank/<domain>/artifact_manifest.json + 产物文件
    living_harness/<domain>/artifact_manifest.json + 产物文件
  val/
    react/<domain>/...
    awm/<domain>/...
    reasoningbank/<domain>/...
    living_harness/<domain>/...
  table.csv
  table.md
```

1. 从当前 `base` split 加载并核验任务，再生成固定的 70/30 train/val split；保存版本和 task ID 清单，不覆盖此前 Exp1 结果。
2. 在三个训练型方法中实现训练后最终产物导出，以及从指定产物加载并冻结的验证路径；确保验证时不触发状态更新或 prompt 再生成。
3. 用少量训练/验证任务做流程检查：不同方法的产物目录互不覆盖，验证 task 不进入训练，冻结前后产物 hash 一致，结果 ID 和 split 一致；同时测量全量运行预算。
4. 固定 split 和配置后，先完成 GPT5.2-base 的三个验证领域，再按 AWM → ReasoningBank → Living-Harness 的方法顺序执行；每个方法按 Airline → Retail → Telecom 逐个完成训练、保存独立 harness、核验并冻结评测。
5. 从验证原始结果按上述公式生成 `table.csv` 和 `table.md`，附每格的成功数/总数、基础设施错误数、产物 hash、代码/数据版本、token/cost 和运行时间，便于审稿人复核。

## 建议运行顺序与命令

GPT5.2-base 的三个验证领域已经完成。训练型方法按表格的行顺序逐个运行；每个领域训练一结束就导出该方法的最终 harness，然后加载该产物评测同领域的验证集。Airline 先运行，可较早检查完整流程。不同 baseline 不同时运行。`run_train_val_baselines.sh` 顺序执行这些步骤。训练若有基础设施错误或缺失 reward，会归档该次尝试并用 2400 秒样本超时从头训练该方法和领域；最终 harness 只从完整训练中导出。验证若有这类错误，则只补跑无效 task；原结果和补跑结果均保留。

```bash
cd code
RUN_ID=train-val-base-300-v1
bash run_train_val.sh --run-id "$RUN_ID" --phase split
TAU2_API_RPM_LIMIT=20 SAMPLE_TIMEOUT_SECONDS=2400 \
  bash run_train_val_baselines.sh "$RUN_ID" 10
```

`run_train_val.sh` 是单作业入口，`run_train_val_baselines.sh` 负责按上述顺序执行及生成最终表格；现有 `test.sh` 仍用于旧的 Exp1 作业。训练和验证结果、各方法各领域的 harness 分开保存在同一 run ID 下。已完成的作业再次调用时会核验并跳过；无法定向补跑的部分作业会保留现场并停止，以便检查。生成表格前，脚本会校验任务 ID、基础设施错误、缺失 reward 和 harness hash。
