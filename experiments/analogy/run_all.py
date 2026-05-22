import os, sys, subprocess, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from experiments.analogy.tee_logger import setup_logger

BASE = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE, 'results')


def run_step(name, script_path):
    print('\n' + '=' * 70)
    print(f'[{name}] Starting...')
    print('=' * 70)
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=BASE,
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f'[{name}] FAILED (return code {result.returncode}, {elapsed:.1f}s)')
        if result.stdout:
            print(f'  stdout (last 2000 chars): {result.stdout[-2000:]}')
        if result.stderr:
            print(f'  stderr (last 2000 chars): {result.stderr[-2000:]}')
        return False
    print(f'[{name}] SUCCESS ({elapsed:.1f}s)')
    return True


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    log_path = os.path.join(RESULT_DIR, 'run_all.log')
    tee = setup_logger(log_path)

    steps = [
        ('Stage 1: Backbone Training',
         os.path.join(BASE, 'stage1_train_backbone.py')),
        ('Stage 2: Probe Training',
         os.path.join(BASE, 'stage2_train_probe.py')),
        ('Evaluation',
         os.path.join(BASE, 'evaluate.py')),
        ('Visualization',
         os.path.join(BASE, 'visualize.py')),
    ]

    for name, script in steps:
        if not run_step(name, script):
            print(f'\nPipeline stopped at [{name}]. Fix errors and retry.')
            tee.close()
            return

    print('\n' + '=' * 70)
    print('Phase 3 Pipeline Complete!')
    print('=' * 70)
    tee.close()


if __name__ == '__main__':
    main()
