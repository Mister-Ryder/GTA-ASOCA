"""Record already-verified evidence; never executes a model or SSH operation."""
from pathlib import Path
import hashlib,json

workspace=Path(__file__).resolve().parents[2]
experiment=workspace/'ASOCA_Extension/fidelity_v2'
ledger=workspace/'autoresearch/classic-261014-gta-fidelity'
accepted_path=experiment/'server_records/unet_acceptance_20261014/SINGLE_RUN_VERIFIED.json'
accepted=json.loads(accepted_path.read_text(encoding='utf-8'))
assert accepted['status']=='verified_unet_fixed_final_and_full_trajectory'
assert accepted['verified_checkpoint_evaluations']==20 and accepted['verified_case_evaluations']==80
assert accepted['pixel_primary_metrics_recomputed']==4
preflight=experiment/'server_records/early_fixed_final_preflight_20261014'
capture=json.loads((preflight/'collection_manifest.json').read_text(encoding='utf-8'))
for row in capture['files']:
    assert hashlib.sha256((preflight/row['name']).read_bytes()).hexdigest()==row['sha256']
new=json.loads((preflight/'38773_EARLY_FINAL_INPUT_RUNTIME_AUDIT_20261014.json').read_text(encoding='utf-8'))
old=json.loads((preflight/'17552_EARLY_FINAL_INPUT_RUNTIME_AUDIT_20261014.json').read_text(encoding='utf-8'))
ready=json.loads((preflight/'17552_EARLY_FINAL_ENV_READY_20261014.json').read_text(encoding='utf-8'))
assert new['cache_CT_inputs']==old['cache_CT_inputs']
assert new['evaluation_sources']==old['evaluation_sources']
assert ready['packages']==new['packages'] and ready['status']=='isolated_eval_runtime_ready'
path=ledger/'handoff.json'
handoff=json.loads(path.read_text(encoding='utf-8'))
handoff.update(current_phase='GTA190/200 joint at latest actual check; UNet200 and20/80 trajectory accepted; fixed-final mirror queued, owned preview waiter pending cancellation repair',
    latest_main_verified_utc='2026-10-14T08:35:04.157356+00:00',latest_main_completed_epochs=190,
    latest_main_stage='joint',latest_baseline_verified_utc=accepted['verified_at_utc'],
    latest_baseline_stage='completed_verified',baseline_internal4_completed_checkpoints=20,
    latest_baseline_checkpoint_observed='final',latest_baseline_metrics_role='fixed-final internal4 development; no superiority or fivefold claim',
    latest_baseline_checkpoint_dice=accepted['final_summary']['overall']['dice']['mean'],
    latest_baseline_checkpoint_assd_mm=accepted['final_summary']['overall']['assd_mm']['mean'],
    latest_baseline_live_handles={'status':'completed; historical PIDs are not live-handle claims'},
    baseline_acceptance_receipt=str(accepted_path.relative_to(workspace)),
    baseline_final_checkpoint_sha256=accepted['final_checkpoint_sha256'],
    early_final_runtime_preflight=str(preflight.relative_to(workspace)),
    early_final_cached_CT_cross_server_actual_bytes_equal=True,
    early_final_isolated_environment_ready=True,
    early_final_mirror_publisher_pid=17636,
    early_final_mirror_publisher_source_sha256='446567f4a964b9c392395cfdb203e6c95e43cffca1d6a576269721dfaf593450',
    early_final_mirror_launch_verified_utc='2026-10-14T08:40:04.539286+00:00',
    shared_atomic_no_replace_actual_probe_passed=True,
    early_fixed_final_inference_started=False,
    early_fixed_final_guard_state='81785 waiting-only controller stop/archive and cancellation lifecycle repair in progress; no evaluator child at last actual state',
    current_goal_turn_progress='Accepted UNet actual20/80 saved predictions and final4 pixel metrics; verified dimensions/wholepatch label rules; aligned isolated old eval runtime and actual4CT bytes; queued CPU final mirror. GTA objective/weights unchanged.',
    user_latest_steering='Explain actual full CT/segmentation patch/label dimensions and the current GT-derived three-head classification rules; keep active run unchanged while assessing label granularity',
    previous_goal_turn_classification='progress: label/Gaussian actual CPU audits and lambda0 CPU initial-state reconstruction completed',
    goal_complete=False)
path.write_text(json.dumps(handoff,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
iterations=ledger/'iterations.tsv'
text=iterations.read_text(encoding='utf-8')
row='1.2\tcommon_recipe_UNet200\tfixed_final_and_trajectory_verified\t0.8184555112537751\t0.7492326552685586\tNA\tNA\tbaseline_only_main_GTA_pending\tActual200x125,20 checkpoints x4=80; all saved prediction hashes/headers, final4 raw-grid Dice+ASSD recomputed; do not pool with exposed8 or claim matched-prior-only ablation\n'
if '\n1.2\t' not in text:iterations.write_text(text.rstrip()+'\n'+row,encoding='utf-8')
print(json.dumps({'status':'local_evidence_recorded','GTA_completed_epochs_at_last_check':190,'UNet_verified_case_evaluations':80,'goal_complete':False}))
