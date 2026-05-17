import subprocess, sys, os

SCRIPTS = [
    ('stage1_train_backbone.py', 'Stage 1: Backbone training'),
    ('stage2_train_probe.py', 'Stage 2: Probe training'),
    ('evaluate.py', 'Evaluation + baselines + ablation'),
    ('visualize.py', 'Visualization'),
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

for script, desc in SCRIPTS:
    print(f'\n{"=" * 60}')
    print(f'{desc}: {script}')
    print(f'{"=" * 60}')
    path = os.path.join(SCRIPT_DIR, script)
    r = subprocess.run([sys.executable, path], cwd=SCRIPT_DIR)
    if r.returncode != 0:
        print(f'ERROR: {script} failed (exit {r.returncode})')
        sys.exit(r.returncode)

print('\nPhase 2b complete.')
