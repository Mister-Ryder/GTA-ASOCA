"""Offline real startup byte fixtures and source parity; no SSH or models."""
import copy
import datetime
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
RECORDS = HERE.parents[1]
ROOT = RECORDS.parent
COLLECTOR = ROOT.parent/'ops/collect_completed_gaussian_control.py'
ADAPTER = ROOT/'ops/verify_completed_gaussian_control.py'

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return json.loads(path.read_text(encoding='utf-8-sig'))

assert sha(HERE/'collect_completed_gaussian_control.before.py') == '4f65c640aaa761cb32cff7cdaeae25b799d3046f8a904f30c6743d845f98c82e'
assert sha(HERE/'verify_completed_gaussian_control.before.py') == 'af05e1c0a06bb4af4c95bc549d310f5575921b3e5a9e719e1ea8f14622f0005f'
assert sha(COLLECTOR) == 'b824932b176c1e9016e494f9f6404391431fdc135fdfee8fc9b7c6799a064b93'
main_path = RECORDS/'gta_training_complete_20261014/results/gta_enhanced200x125/training/config.json'
assert sha(main_path) == 'b61358e8e4e3be2b8b1a194815c50de808672a76d77c450daa9ea55ddfca598b'
main = read(main_path)
specs = {
    'lambda0': ('lambda0_actual_launch_20261014','expected_lambda0_config.json','expected_control_config_sha256',
                'ACTUAL_CONTROL_THETA0.json','run_gaussian_lambda0_control.py',
                '31777dcd11f7d08579038a322a39874eb4b4244db3aba18fb78f01cecd4e5040'),
    'B': ('B_formal_launch_20261014/actual_startup','expected_B_config.json','expected_B_config_sha256',
          'ACTUAL_B_THETA0.json','run_gaussian_region_balance_control.py',
          'ace0561942714916fe35e4717a1c3df2736aa85b44e63bf978ea9b9fa8da1f1b'),
}
fixtures = {}
for kind,(folder,expected_name,expected_key,theta_name,producer_name,producer_sha) in specs.items():
    base = RECORDS/folder
    protocol = read(base/'protocol.json')
    expected_path, actual_path = base/expected_name, base/'actual_training_config.json'
    expected, actual, theta = read(expected_path), read(actual_path), read(base/theta_name)
    assert expected == actual
    assert protocol[expected_key] == sha(expected_path)
    assert sha(expected_path) != sha(actual_path)
    assert expected_path.read_bytes() == actual_path.read_bytes()+b'\n'
    assert sha(ROOT/'ops'/producer_name) == producer_sha == protocol['controller_sha256']
    reference = copy.deepcopy(main); reference['output'] = actual['output']
    if kind == 'lambda0':
        reference['gaussian_weight'] = 0.; reference['gaussian_schedule']['base'] = 0.
        assert theta['config_sha256'] == sha(actual_path)
        assert theta['actual_original_main_epoch0_state_verified'] is False
        binding = {'config_sha256':theta['config_sha256']}
    else:
        reference.update(source_sha256=actual['source_sha256'],source_files_sha256=actual['source_files_sha256'])
        assert [k for k in main['source_files_sha256'] if main['source_files_sha256'][k] != actual['source_files_sha256'][k]] == ['losses.py']
        binding = theta['actual_config_binding']
        assert binding['actual_config_sha256'] == sha(actual_path)
        assert binding['actual_config_source_sha256'] == actual['source_sha256']
        assert binding['B_protocol_sha256'] == protocol['B_protocol_sha256']
        assert theta['historical_main_theta0_verified'] is False
    assert actual == reference
    latest = RECORDS/'dual_control_progress_snapshots/20261015T062955_404096Z'/('17552.config.json' if kind == 'lambda0' else '38773.config.json')
    assert sha(latest) == sha(actual_path), 'Actual terminal snapshot config differs from launch bytes'
    fixtures[kind] = {'startup_folder':str(base),'expected_bytes_sha256':sha(expected_path),
        'actual_bytes_sha256':sha(actual_path),'protocol_expected_SHA_binds_expected_file':True,
        'expected_parsed_object_equals_actual_and_exact_allowed_A_delta':True,
        'byte_difference':'Expected file is actual file plus exactly one trailing LF byte',
        'actual_bytes_bound_by_readonly_startup_receipt':binding,
        'latest_actual_terminal_snapshot_config_bytes_unchanged':True,
        'producer_source_sha256':producer_sha}

diff = []
for name,current in [('collect_completed_gaussian_control',COLLECTOR),('verify_completed_gaussian_control',ADAPTER)]:
    before = HERE/(name+'.before.py')
    diff.extend(difflib.unified_diff(before.read_text(encoding='utf-8').splitlines(keepends=True),
        current.read_text(encoding='utf-8').splitlines(keepends=True),fromfile=str(before),tofile=str(current)))
(HERE/'incremental_source_diff.patch').write_text(''.join(diff),encoding='utf-8',newline='\n')
assert "COLLECTOR_SHA='"+sha(COLLECTOR)+"'" in ADAPTER.read_text(encoding='utf-8')
plans=[]
for label,helper in [('collector',COLLECTOR),('cpu_adapter',ADAPTER)]:
    for kind in ('lambda0','B'):
        result = subprocess.run([sys.executable,'-B',str(helper),'--kind',kind,'--plan'],capture_output=True,timeout=120)
        assert result.returncode == 0, result.stderr.decode('utf-8',errors='replace')
        json.loads(result.stdout.decode('utf-8-sig'))
        name=label+'_'+kind+'_plan'
        (HERE/(name+'.stdout.json')).write_bytes(result.stdout)
        (HERE/(name+'.stderr.txt')).write_bytes(result.stderr)
        plans.append({'helper':label,'kind':kind,'returncode':result.returncode})
audit = {'schema_version':1,'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'status':'offline_real_startup_byte_parity_and_four_plans_passed_actual_complete_capture_pending',
    'repair':'Bind expected_* protocol SHA to expected file, retain strict expected-object==actual-object and exact A recipe delta; bind actual bytes to theta0 receipt and existing trajectory identity. CPU adapter collector pin updated only.',
    'real_startup_fixtures':fixtures,
    'remaining_source_parity_review':[
        'CONTROL_COMPLETE fields/status and key extension: pinned producer source AST verified in prior round.',
        'POSTFIT_VERIFIED keys status/root/training/count20/count80/history/training_complete/trajectory/curves/cases and all five curve artifact names match verify_postfit.postflight.',
        'lambda0 cpu_preflight/CONTROL_PREFLIGHT.json, completion postfit_verified_sha256/formal_training_complete_sha256/verified_case_inferences and theta config_sha256 match producer.',
        'B ACTUAL_B_CONFIG_BINDING.json, actual_B_theta0_receipt_sha256/B_postfit_verified_sha256/training_complete_sha256/verified counts/A_canonical_unchanged and theta embedded binding match producer.',
        'Trajectory config SHA is actual bytes; source, IDs, split, epochs9..189/final199, evaluator map and checkpoint names match both producer preflights and evaluate_trajectory.',
        'Curve sources use actual config/history/validation/summary paths and optional gradient_probes; all are within exact control inventory. Five artifact hashes come from actual POSTFIT_VERIFIED.',
        'Collector paths map OLD controller records separately from OLD logical training root; evaluator/source/freeze are pinned. No POSTFIT_STATUS or standalone EVALUATION_COMPLETE invented.',
        'CPU adapter passes B source through sealed context and re-runs exact actual-config checker; no expected file SHA substituted into actual binding.',
        'No further analogous byte-versus-object mismatch or producer field/path mismatch identified by source inspection. Actual postfit bytes not yet locally available: ROOT must retry captures and actual CPU acceptance.'
    ],'collector_sha256':sha(COLLECTOR),'cpu_adapter_sha256':sha(ADAPTER),'plans':plans,
    'SSH_used':False,'models_or_weights_loaded':False,'acceptance_run':False}
(HERE/'repair_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')
print(json.dumps({k:audit[k] for k in ('status','collector_sha256','cpu_adapter_sha256','plans')},indent=2))
