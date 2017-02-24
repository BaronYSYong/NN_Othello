"""
Reference:
https://en.wikipedia.org/wiki/Q-learning
http://mnemstudio.org/path-finding-q-learning-tutorial.htm

Q[state, action] = (1-alpha)*Q[state, action] + alpha * (R[state, action]) + gamma * max_Q[next state, all action])
"""

import numpy as np


if __name__ == "__main__":
	R = np.array(
	[[-1, -1, -1, -1,  0,  -1],
    [-1, -1, -1,  0, -1, 100],
    [-1, -1, -1,  0, -1,  -1],
    [-1,  0,  0, -1,  0,  -1],
    [ 0, -1, -1,  0, -1, 100],
    [-1,  0, -1, -1,  0, 100]]).astype("float32")
    
	Q = np.zeros_like(R)
	print R
	print Q
