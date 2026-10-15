"""Offline source/diff/--plan audit; never imports producers or runs acceptance."""
import ast
import datetime
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
FIDELITY = HERE.parent.parent
COLLECTOR = FIDELITY.parent/'ops/collect_completed_gaussian_control.py'
ADAPTER = FIDELITY/'ops/verify_completed_gaussian_control.py'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def literal_keys(node):
    assert isinstance(node, ast.Dict)
    return {ast.literal_eval(k) for k in node.keys}


producer_specs = {
    'lambda0': ('run_gaussian_lambda0_control.py',
                '31777dcd11f7d08579038a322a39874eb4b4244db3aba18fb78f01cecd4e5040',
                'formal_control_authorized_recipe'),
    'B': ('run_gaussian_region_balance_control.py',
          'ace0561942714916fe35e4717a1c3df2736aa85b44e63bf978ea9b9fa8da1f1b',
          'actual_ROOT_decision_authorized_formal_B'),
}
producer_audit = {}
for kind, (name, expected_sha, launch_status) in producer_specs.items():
    path = FIDELITY/'ops'/name
    assert sha(path) == expected_sha, name+' changed'
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    plan = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'plan')
    plan_return = next(n for n in plan.body if isinstance(n, ast.Return))
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    assignments = {}
    for node in ast.walk(run):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ('protocol', 'complete'):
                assert node.targets[0].id not in assignments, 'Ambiguous producer assignment'
                assignments[node.targets[0].id] = node
    protocol = assignments['protocol']
    complete = assignments['complete']
    assert isinstance(protocol.value, ast.BinOp) and isinstance(protocol.value.op, ast.BitOr)
    assert isinstance(protocol.value.left, ast.Call) and isinstance(protocol.value.left.func, ast.Name)
    assert protocol.value.left.func.id == 'plan'
    assert isinstance(complete.value, ast.BinOp) and isinstance(complete.value.op, ast.BitOr)
    assert isinstance(complete.value.left, ast.Name) and complete.value.left.id == 'protocol'
    launch_keys = literal_keys(plan_return.value) | literal_keys(protocol.value.right)
    final_keys = literal_keys(complete.value.right)
    overwritten = launch_keys & final_keys
    assert overwritten == {'status'}, 'Unexpected overwritten launch fields: '+str(overwritten)
    protocol_values = dict(zip((ast.literal_eval(k) for k in protocol.value.right.keys), protocol.value.right.values))
    complete_values = dict(zip((ast.literal_eval(k) for k in complete.value.right.keys), complete.value.right.values))
    assert ast.literal_eval(protocol_values['status']) == launch_status
    assert ast.literal_eval(complete_values['status']) == 'completed'
    assert 'completed_at_utc' not in launch_keys and 'children' not in launch_keys
    producer_audit[kind] = {
        'source': str(path), 'sha256': expected_sha,
        'protocol_assignment_line': protocol.lineno,
        'completion_assignment_line': complete.lineno,
        'completion_overwrites_launch_keys': sorted(overwritten),
        'required_status_transition': [launch_status, 'completed'],
        'new_completion_fields_not_ignored_launch_fields': sorted(final_keys-launch_keys),
    }

original_collector = HERE/'collect_completed_gaussian_control.original.py'
original_adapter = HERE/'verify_completed_gaussian_control.original.py'
assert sha(original_collector) == 'ec612fd2e1ca017dbca81cbbafa48df136617d91a1d9c7147aa96db1ef24735f'
assert sha(original_adapter) == 'f5c9ea6e23247989ddc3311be0d88e50f4e2cbd39fb542e0d90c19eb1a147666'
assert sha(COLLECTOR) == '4f65c640aaa761cb32cff7cdaeae25b799d3046f8a904f30c6743d845f98c82e'
assert "COLLECTOR_SHA='"+sha(COLLECTOR)+"'" in ADAPTER.read_text(encoding='utf-8')
diff = []
for original, current in ((original_collector, COLLECTOR), (original_adapter, ADAPTER)):
    diff.extend(difflib.unified_diff(original.read_text(encoding='utf-8').splitlines(keepends=True),
        current.read_text(encoding='utf-8').splitlines(keepends=True),
        fromfile=str(original), tofile=str(current)))
(HERE/'actual_source_diff.patch').write_text(''.join(diff), encoding='utf-8', newline='\n')

plans = []
for label, helper in (('collector', COLLECTOR), ('cpu_adapter', ADAPTER)):
    for kind in ('lambda0', 'B'):
        argv = [sys.executable, '-B', str(helper), '--kind', kind, '--plan']
        result = subprocess.run(argv, capture_output=True, timeout=120, check=False)
        assert result.returncode == 0, label+' '+kind+' --plan failed: '+result.stderr.decode('utf-8', errors='replace')
        value = json.loads(result.stdout.decode('utf-8-sig'))
        name = label+'_'+kind+'_plan'
        (HERE/(name+'.stdout.json')).write_bytes(result.stdout)
        (HERE/(name+'.stderr.txt')).write_bytes(result.stderr)
        plans.append({'helper': label, 'kind': kind, 'returncode': result.returncode,
                      'stdout_sha256': sha(HERE/(name+'.stdout.json')), 'plan': value})

audit = {
    'schema_version': 1,
    'status': 'offline_source_contract_and_four_plans_verified_actual_controls_not_accepted_here',
    'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'reason': 'The completed producers intentionally replace launch status. The old collector compared status as immutable and rejected valid completion before downloading any records.',
    'repair_scope': 'Only unfrozen local collector status extension check and CPU adapter COLLECTOR_SHA. No frozen or deployed producer, model, evaluator, metric, method config or live process modified.',
    'allowed_overwritten_launch_fields': ['status'],
    'gate_policy': 'Require exact kind-specific launch status and completed terminal status; require every launch key present in completion and equal except that explicit status transition. Existing released STATUS, receipt SHA and postfit20/80 gates retained.',
    'producer_source_AST_audit': producer_audit,
    'collector': {'before_sha256': sha(original_collector), 'after_sha256': sha(COLLECTOR)},
    'cpu_adapter': {'before_sha256': sha(original_adapter), 'after_sha256': sha(ADAPTER)},
    'plans': plans,
    'SSH_performed': False, 'models_loaded': False, 'inference_performed': False,
    'actual_capture_retry_required': True, 'actual_CPU_acceptance_required': True,
}
(HERE/'repair_audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False)+'\n', encoding='utf-8', newline='\n')
print(json.dumps({'status': audit['status'], 'collector_sha256': sha(COLLECTOR),
                  'cpu_adapter_sha256': sha(ADAPTER), 'four_plan_returncodes': [p['returncode'] for p in plans]}, indent=2))
