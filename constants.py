HEADLESS = False
LOG = False

WIDTH, HEIGHT = 800, 600


PADDLE_SIZE = 75
PADDLE_WIDTH = 12
PADDLE_X = 25

BALL_RADIUS = 10

PADDLE_RANGE = HEIGHT - PADDLE_SIZE
BALLX_RANGE = WIDTH - PADDLE_X*2
BALLY_RANGE = HEIGHT 

ALLOW_INPUT = False


EPISODE_LENGTH = 30 * 3 #time steps / frames
BATCHES = 33 # how many episodes to collect before optimizing

MATCHES = None # how many matches to play before stopping, none for infinite