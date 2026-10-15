ASOCA coronary extension of the archived GTAirway v2 method

最新范围决定：2026-10-14（Asia/Shanghai）。先审查方法忠实性及恢复
预训练/阶段训练的可行性。waiting coordinator PID102978 已SIGSTOP，
尚未启动的四项自身fold0消融全部pending暂停；native PID54628保留收尾。
当前实况以 development/method_fidelity_hold.json 为准；不自动继续旧四项。
本README与 docs/method_fidelity_review_20261014.txt 覆盖此前scope_decision
中的继续消融安排及旧五折/30任务安排，原结果/方案/失败记录保留溯源。

Source version: GTAirway main commit 0254c339c910ba17b7ddd682215e1b946c95565f.
The original GTAirway directory is not modified. Historical modules and their
hashes/provenance are retained in code/historical. Adapted scripts are in code/.

Task definition
  ASOCA has 40 annotated cases (20 Normal,20 Diseased), labels0/1 for background
  and coronary lumen. The other20 challenge images have no public reference
  masks. This is binary coronary segmentation, not multiclass whole-heart work.
  Normal/Diseased is a cohort variable. Auxiliary depth/structure/location
  categories are geometric training targets, not coronary clinical classes.

Shared method
  Preserve v2 three-head prior encoder, scale-specific guidance, multiscale
  decoder fusion, cumulative radius-aware Gaussian auxiliary and historical
  power-Dice/focal/guidance objectives. Add optional half/quarter decoder DS;
  upsample logits to the original patch target to retain thin-vessel supervision.
  当前已完成 ASOCA GTA fold0 的 no_ds=False，已经加了标准分割深监督。
  主/半/四分之一输出为96³/48³/24³；loss weights [1,.5,.25]除以1.75，
  实际相对权重为4/7、2/7、1/7。这是分割目标内部的尺度权重，不是
  全部属性/指导/Gaussian多任务目标的权重。推理只使用主输出。
  该主运行已完成200epochs×125更新；原肺气道两版本代码没有因此改动。
  当前ASOCA确未加载预训练，分类器与分割网络从头联合训练；不能据此
  断言历史权重不存在。备份已找到BAS599、classifier improved/hybrid100
  和GTA联合350候选，兼容性及是否原best尚未验证。下一优先级是核验
  候选权重并审查恢复预训练+分阶段训练，不称随机联合训练为已预训练。
  主体架构保留；当前新增DS仅在decoder，增强forward未调用encoder fusion。
  原base预训练曾使用两个seg输出，不能说原流程完全无encoder/decoder监督。
  Details: docs/method_audit.txt and docs/method_fidelity_review_20261014.txt.

Data and evaluation
  Fixed seed20261013 retains five patient-level stratified fold definitions.
  当前仅使用fold0：32训练、8方法开发验证（正常4+患病4）。验证病例为
  Diseased_1/15/18/7、Normal_16/19/2/8；已用于方法诊断和选择，不是
  独立final test或官方隐藏20例。固定final checkpoint避免挑最佳epoch，
  但不能消除反复用同一开发集选方法的偏差；方法确定后再冻结最终评估。
  LPS orientation,whole CT field,0.5mm isotropic resampling,intensity clip
  [-200,1000] then(image-400)/600. Inference only uses CT. No GT-derived image ROI.
  Auxiliary targets are training-only. Actual original-space masks are used for
  development metrics. Fixed final checkpoints prevent heldout epoch selection.
  Both conventional surfaceHD95 and ASOCA foreground-pointHD95 are named.
  Empty predictions remain failures, with undefined distance means made explicit.

Comparison queue
  native nnU-Net fold0当前运行保留收尾；其后不自动启动自身消融。
  此前准备的gta_no_ds→gta_no_prior→gta_no_gaussian→gta_no_balance全部
  pending暂停，待方法忠实性审查与预训练/阶段恢复方案明确后再安排。
  gta_no_ds关闭标准分割深监督；no_prior去属性指导；no_gaussian去Gaussian
  分支及相应损失；no_balance同时去属性平衡采样与对应样本损失权重。
  no_balance检验联合平衡机制，不能单独归因于采样或loss weights。
  已完成GTA、U-Net、Swin UNETR同8例结果保留为阶段参考。
  当前GTA Dice80.01%、clDice83.67%只属于fold0、N=8的方法开发验证。
  nnU-Net训练pseudo-Dice不能与原生空间8例评价指标混排。
  延期：fold1–4、未启动外部复现、unet_no_ds及U-Net+clDice恢复。
  U-Net+clDice fold0 epoch153附近非有限梯度失败保留；恢复标记为
  deferred_by_user，不把失败过程填成已完成结果。
  Custom runs:200epochs x125updates=25,000updates,96cubed patches,batch2.
  Native nnU-Net100epochs x250updates=25,000updates; its own planning,optimizer,
  patch size,batch size and augmentations remain native and are recorded.
  This is a shortened native budget, not the original1000epoch nnU-Net default.
  Equal update counts do not imply equal GPU hours or equal effective data volume.
  阶段性消融只回答当前开发验证集上的设计问题，不代表五折泛化结果。
  原aggregate仍按其定义标记fivefold incomplete；开发队列完成不会
  改写为完整五折已完成。

Layout
  code/  runnable,versioned experiment code
  ops/   SSH transfer/control helpers;password is prompted and never stored
  docs/  audited dataset,protocol,AEI narrative and method adaptation decisions
  pilot_cache/ one true-data case for smoke only;never a publication experiment
  server_records/ captured remote manifests/logs
  development/coordinator.py 新控制器（位于主code/之外）
  development/development_protocol.json 新范围与来源记录
  results/development_queue_status.json 开发队列记录（当前waiting控制器冻结）
  development/method_fidelity_hold.json 本轮暂停与保留native收尾的当前实况
  自身结果沿用results/<ablation>/fold0/training和evaluation。
  原protocol_identity、JOB_COMPLETE和源码快照继续保留。

Server experiment root
  /root/autodl-tmp/ASOCA_GTAirway_20261013
  env/bin/python is the dedicated Python3.11 + PyTorch2.7.1 CUDA12.8 environment.
  raw/CompressedVersion.zip is the SHA256-verified original dataset archive.
  raw/CompressedVersion is the extracted source.
  cache/manifest.json contains frozen folds,physical grids and source hashes.
  results/<method>/foldN/training and evaluation retain configs,weights and cases.
  logs/formal.log and results/queue_status.json preserve the historical queue.
  旧父调度器及waiting开发控制器均已冻结；当前范围以method_fidelity_hold.json为准。

Execution order
  ops/bootstrap.sh installs and verifies the isolated GPU environment.
  ops/smoke.sh verifies loss bounds,old/new model equivalence,CUDA gradients and
  actual96cubed,batch2 training steps for all custom models.
  ops/formal.sh prepared all40 cases and launched the historical full queue.
  当前开发入口为development/coordinator.py，不据旧入口自动展开五折。
  Only one training process runs on the GPU.
  Code resume checks frozen manifest and source hashes;do not edit active source.
  A failed job is logged explicitly;the queue does not relabel substitute models.

Paper angle
  Airway and coronary experiments jointly address sparse branching tubular
  reconstruction. Explicit structure knowledge should be described as the
  representation and use of geometric attributes. Cross-anatomy experiments do
  not alone prove AEI scope suitability or downstream simulation/clinical benefit.
  See docs/ASOCA_experiment_protocol_and_AEI_narrative.txt for verified sources.
  其早期五折执行建议已被本次小范围开发决定覆盖。
  docs/ASOCA_literature_comparison_20261014.txt保留作者报告值；完整五折、
  固定测试或官方盲测是不同协议参照，不能与当前8例80.01%直接相减后
  宣称严格同条件排名。暂不新增逐篇外部复现。
