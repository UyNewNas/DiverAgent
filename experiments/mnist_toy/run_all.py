import subprocess
import sys
import os

SCRIPTS_DIR = os.path.dirname(__file__)


def run_step(name, script):
    print(f'\n{"#"*60}')
    print(f'# STEP: {name}')
    print(f'{"#"*60}')
    result = subprocess.run(
        [sys.executable, os.path.join(SCRIPTS_DIR, script)],
        cwd=SCRIPTS_DIR,
    )
    if result.returncode != 0:
        print(f'\nERROR: {name} failed with code {result.returncode}')
        sys.exit(result.returncode)
    print(f'\n{name} completed successfully.')


def main():
    print('=' * 60)
    print('CBDP Phase 0: MNIST Toy Experiment')
    print('=' * 60)

    run_step('1/3 Training Convergent Backbone', 'train_backbone.py')
    run_step('2/3 Training Divergent Probe', 'train_probe.py')
    run_step('3/3 Evaluation', 'evaluate.py')

    print('\n' + '=' * 60)
    print('Phase 0 complete!')
    print('Run visualize.py to see generated samples.')
    print('=' * 60)


if __name__ == '__main__':
    main()
