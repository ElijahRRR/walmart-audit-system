"""评测 / 采样 / 诊断脚本 (非自动化测试)。

与 tests/ 的分工:
  tests/  — pytest 回归测试, CI/提交前跑, 无网络无外部依赖, 秒级
  evals/  — 带 __main__ 的一次性脚本, 手工跑, 依赖真实 DB / DMIT / LLM 配额, 分钟~小时级

跑法: python3 -m evals.<脚本名> [args]  (每个脚本 docstring 顶部有具体用法)

常用:
  evals.eval_l3_precision     L3 判定精度回测 (新备用模型进路由链前的质量门禁)
  evals.l1_accuracy_eval      L1 类目判定精度
  evals.dump_pipeline_v2      导出全链路判定明细 (上面几个 eval 的输入)
  evals.diagnose_l1           单个 ASIN 的 L1 判定过程诊断
"""
