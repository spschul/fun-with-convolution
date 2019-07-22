import numpy as np
from scipy.signal import convolve2d

def main():
    image = np.arange(7680 * 4 * 4320 * 4, dtype=np.int32).reshape((7680 * 4, 4320 * 4)) % 100
    ker = np.arange(9, dtype=np.int32).reshape((3, 3))

    convolve2d(image, ker)

if __name__ == '__main__':
    main()
