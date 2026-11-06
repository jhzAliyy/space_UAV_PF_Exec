# workflows · 用途索引

> 全部工作流均为 `workflow_dispatch`；训练/评估/核验在 GitHub Actions 上运行。
> 信道表从 release `chan-tables` 下载；凭据只使用仓库 Secrets。

| 工作流 | 用途 |
|---|---|
| `theory-audit.yml` | 理论篇 D1–D3 在 n3~n100 的数值核验（numpy + scipy） |
| `eval-now.yml` / `eval-one.yml` | 单批/单格评估 |
| `eval-real.yml` / `eval-real2.yml` | 真实信息退化线评估 |
| `eval-scan.yml` / `eval-sota.yml` | 扫描/外部基线评估 |
| `eval-seedfill.yml` | 补种子评估 |
| `train-a2a.yml` | 场参数化（a2a）对照训练 |
| `train-real.yml` / `train-real2.yml` / `train-real3.yml` | 真实信息退化线训练 |
| `train-scale.yml` / `train-seedfill.yml` | 规模/补种子训练 |
| `speed12sota.yml` | 外部基线测速 |
| `pfg_eval.yml` / `transfer.yml` | 历史评估/迁移辅助 |

`theory-audit.yml` 的 checkout 使用 `${{ github.ref_name }}`，在 main 上直接
运行；其余工作流保留为设施，按需手动触发。
