"""NOT_MODEL_RESULTS: adversarial ownership-cancellation fixtures; no OS child."""
import argparse
import json
from pathlib import Path
import signal
import subprocess
from unittest.mock import patch

import run_early_fixed_final_eval as guard


class MockOwnedChild:
    pid = 424242

    def __init__(self, cancellation, events):
        self.cancellation = cancellation
        self.events = events
        self.returncode = None
        self.waits = 0

    def poll(self):
        return self.returncode

    def wait(self, timeout):
        self.waits += 1
        self.events.append(('wait', timeout))
        if self.waits <= 2:
            self.cancellation.request(signal.SIGINT)
            raise subprocess.TimeoutExpired('NOT_MODEL_RESULTS_mock_child', timeout)
        self.returncode = -9
        self.events.append(('owned_exit_confirmed', self.returncode))
        return self.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    root = Path(__file__).resolve().parent.parent
    assert output.is_relative_to((root/'server_records/tool_fixtures').resolve()) and not output.exists()
    output.mkdir(parents=True)
    events = []; cancellation = guard.OwnedCancellation(); child = None; lease_held = True

    def adversarial_constructor():
        created = MockOwnedChild(cancellation, events)
        events.append(('child_created_before_assignment', created.pid))
        cancellation.request(signal.SIGTERM)
        assert child is None  # Exact vulnerable assignment window.
        events.append(('constructor_returned_without_raising_signal', created.pid))
        return created

    cancellation.launching = True
    try:
        child = adversarial_constructor()
        owned_pgid = child.pid
    finally:
        cancellation.launching = False
    assert child is not None and owned_pgid == child.pid
    try:
        cancellation.raise_pending()
    except KeyboardInterrupt:
        events.append(('cancellation_raised_after_child_owned', child.pid))
    else:
        raise AssertionError('Deferred launch cancellation was lost')
    assert cancellation.cleaning is True  # Repeated signals cannot interrupt unwind entry.
    delivered = []

    def kill_owned(pgid, signum):
        assert pgid == child.pid and lease_held
        delivered.append((pgid, int(signum)))
        cancellation.request(signal.SIGTERM)  # Repeated cancellation during cleanup.

    def progress():
        assert lease_held and child.poll() is None
        events.append(('lease_retained_while_owned_child_live', child.pid))

    with patch.object(guard.os, 'getpgid', return_value=child.pid, create=True), patch.object(guard.os, 'killpg', side_effect=kill_owned, create=True), patch.object(guard.signal, 'SIGKILL', 9, create=True):
        guard.cleanup_owned_child(child, progress)
    assert child.poll() is not None and delivered == [(child.pid, int(signal.SIGTERM)), (child.pid, 9)]
    assert cancellation.pending == [int(signal.SIGTERM), int(signal.SIGTERM), int(signal.SIGINT), int(signal.SIGTERM), int(signal.SIGINT)]
    assert lease_held
    lease_held = False
    events.append(('GPU_lease_released_only_after_owned_exit', child.pid))

    # PGID inspection failure cannot justify either signaling an unrelated
    # group or releasing the lease before the owned Popen object reports exit.
    fallback_cancel = guard.OwnedCancellation(); fallback_cancel.cleaning = True
    fallback = MockOwnedChild(fallback_cancel, events)
    lease_held = True
    with patch.object(guard.os, 'getpgid', side_effect=PermissionError('NOT_MODEL_RESULTS'), create=True), patch.object(guard.os, 'killpg', create=True) as signals:
        guard.cleanup_owned_child(fallback, lambda: events.append(('fallback_owned_wait_with_lease_held', lease_held)))
        assert not signals.called and fallback.poll() is not None and lease_held
    lease_held = False
    report = {'status':'passed', 'role':'NOT_MODEL_RESULTS', 'controller_source_sha256':guard.sha(Path(guard.__file__)),
              'fixture_source_sha256':guard.sha(Path(__file__)), 'checks':[
                  'SIGTERM_before_Popen_assignment_deferred_until_PID_PGID_owned',
                  'repeated_cancel_during_TERM_KILL_wait_cannot_interrupt_cleanup',
                  'GPU_lease_retained_until_owned_child_exit_confirmed',
                  'PGID_inspection_failure_sends_no_unrelated_group_signal_and_keeps_lease_until_exit'],
              'events':events, 'actual_OS_process_started':False, 'actual_OS_signal_sent':False,
              'server_connections':0, 'model_inference_or_training_started':False,
              'not_executed':['Linux subprocess and OS signal delivery', 'GPU or server evaluation']}
    (output/'fixture_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
