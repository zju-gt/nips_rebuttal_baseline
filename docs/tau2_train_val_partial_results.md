# tau2-bench 70/30 train/val 已完成结果

运行 ID：`train-val-base-300-v1`。以下数值为固定验证集上的任务成功率。

| Method | Airline | Retail | Telecom | Avg |
| --- | ---: | ---: | ---: | ---: |
| GPT5.2-base | 66.67% | 76.47% | 58.82% | 67.32% |
| AWM | 46.67% | 79.41% | 44.12% | 56.73% |

验证集规模分别为 Airline 15、Retail 34、Telecom 34 条任务。`Avg` 为三个领域成功率的等权平均。上述六个验证作业均无基础设施错误或缺失 reward。

结果摘要位于 `code/results/train-val-base-300-v1/val/{react,awm}/{airline,retail,telecom}/summary.json`。
