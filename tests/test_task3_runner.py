import os
from pathlib import Path
import shutil
import subprocess
import sys


def test_sequential_runner_stops_on_error_and_skips_completed_stages(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/task3_finish.sh"
    (tmp_path / "scripts").mkdir()
    shutil.copy(script, tmp_path / "scripts/task3_finish.sh")
    for name in [
        "data/task3/environment.json",
        "data/task3/sources.json",
        "data/task3_downloads/math_questions.json",
        "ATML-PA3-Data/README.MD",
    ]:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}")
    for kind in ("math", "mathvista"):
        for suffix in (
            "candidates.jsonl",
            "labels.jsonl",
            "splits.json",
            "orders.json",
        ):
            (tmp_path / f"data/task3/{kind}_{suffix}").write_text("{}")
    binary = tmp_path / "bin"
    binary.mkdir()
    shim = binary / "python"
    shim.write_text(
        "#!"
        + sys.executable
        + "\n"
        + """
import json, os, sys
from pathlib import Path
if sys.argv[1:] == ['-']:
    exec(sys.stdin.read())
    raise SystemExit(0)
command = sys.argv[3] if sys.argv[2] == 'pa3.cli' else 'examples'
with Path('events.txt').open('a') as f: f.write(command + '\\n')
if os.environ.get('FAIL_STAGE') == command: raise SystemExit(1)
if command == 'task3-offline':
    names = [f'results/task3/offline/{kind}_{suffix}'
        for kind in ('math','mathvista')
        for suffix in ('summary.csv','per_question.jsonl','question_means.csv',
                       'accuracy_differences.json','signals.jsonl','rollout_scores.jsonl','trajectories.jsonl')]
elif command == 'task3-adaptive':
    names = [f'results/task3/adaptive/{kind}_{suffix}'
        for kind in ('math','mathvista')
        for suffix in ('frozen_policies.json','development_grid.json','adaptive_summary.json',
                       'agreement_heldout.jsonl','quality_heldout.jsonl')]
elif command == 'task3-live':
    names = ['results/task3/live/' + name for name in ('calls.jsonl','per_question.jsonl','environment.json')]
else: names = []
for name in names:
    p = Path(name); p.parent.mkdir(parents=True, exist_ok=True); p.write_text('{}')
if command == 'audit':
    Path('results/task3/completion-audit.json').write_text(json.dumps({'checks':[
        {'requirement':'Task 3 synthetic orchestration check', 'passed':True}]}))
"""
    )
    shim.chmod(0o755)
    env = {**os.environ, "PATH": str(binary) + os.pathsep + os.environ["PATH"]}
    command = ["bash", "scripts/task3_finish.sh"]
    failed = subprocess.run(
        command,
        cwd=tmp_path,
        env={**env, "FAIL_STAGE": "task3-adaptive"},
        capture_output=True,
        text=True,
    )
    assert failed.returncode != 0
    assert (tmp_path / "events.txt").read_text().splitlines() == [
        "task3-offline",
        "task3-adaptive",
    ]
    assert not (tmp_path / "results/task3/runner.lock").exists()
    resumed = subprocess.run(
        command, cwd=tmp_path, env=env, capture_output=True, text=True
    )
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    events = (tmp_path / "events.txt").read_text().splitlines()
    assert events == [
        "task3-offline",
        "task3-adaptive",
        "task3-adaptive",
        "task3-live",
        "examples",
        "plots",
        "audit",
    ]
    again = subprocess.run(
        command, cwd=tmp_path, env=env, capture_output=True, text=True
    )
    assert again.returncode == 0, again.stdout + again.stderr
    assert (tmp_path / "events.txt").read_text().splitlines()[len(events) :] == [
        "examples",
        "plots",
        "audit",
    ]
