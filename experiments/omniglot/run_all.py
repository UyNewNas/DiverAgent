import subprocess
import sys
import os

SCRIPTS = [
    'stage1_train_backbone.py',
    'stage2_train_probe.py',
    'evaluate.py',
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

for script in SCRIPTS:
    path = os.path.join(SCRIPT_DIR, script)
    print(f'\n{"=" * 60}')
    print(f'Running: {script}')
    print(f'{"=" * 60}')
    result = subprocess.run([sys.executable, path], cwd=SCRIPT_DIR)
    if result.returncode != 0:
        print(f'ERROR: {script} failed with exit code {result.returncode}')
        sys.exit(result.returncode)

print('\nPhase 1 complete.')
