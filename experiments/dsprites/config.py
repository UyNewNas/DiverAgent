IMAGE_SIZE = 64
LATENT_DIM = 128
OBJECT_DIM = 64
ATTR_DIM = 64
K_OUTPUTS = 8
BATCH_SIZE = 32

NUM_SHAPES = 3
NUM_COLORS = 6
SHAPES = ['square', 'ellipse', 'heart']
COLORS = {
    'red': (1.0, 0.0, 0.0),
    'green': (0.0, 1.0, 0.0),
    'blue': (0.0, 0.0, 1.0),
    'yellow': (1.0, 1.0, 0.0),
    'magenta': (1.0, 0.0, 1.0),
    'cyan': (0.0, 1.0, 1.0),
}
TRAIN_COLORS = ['red', 'green', 'blue', 'yellow']
TEST_COLORS = ['magenta', 'cyan']

BACKBONE_EPOCHS = 80
BACKBONE_LR = 1e-3
PROBE_EPOCHS = 15
PROBE_LR = 1e-3
LAMBDA_DIVERSITY = 0.15
LAMBDA_PLAUSIBILITY = 0.8
LAMBDA_NOVELTY = 0.5
NOVELTY_MEMORY_SIZE = 3000
DEVICE = 'cuda'
