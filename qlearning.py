"""
Reference:
https://en.wikipedia.org/wiki/Q-learning
http://mnemstudio.org/path-finding-q-learning-tutorial.htm

Q[state, action] = (1-alpha)*Q[state, action] + alpha * (R[state, action]) + gamma * max_Q[next state, all action])
"""

import numpy as np

def update_q(state, next_state, action, alpha = 1.0, gamma = 0.8):
    Q[state, action] = (1-alpha)*Q[state, action] + alpha * (R[state, action]) + gamma * max(Q[next_state, :])
    return Q[state, action]

R = np.array(
[[-1, -1, -1, -1,  0,  -1],
[-1, -1, -1,  0, -1, 100],
[-1, -1, -1,  0, -1,  -1],
[-1,  0,  0, -1,  0,  -1],
[ 0, -1, -1,  0, -1, 100],
[-1,  0, -1, -1,  0, 100]]).astype("float32")

Q = np.zeros_like(R)

#~ if __name__ == "__main__":
#~ for row in range(len((Q))):
    #~ for column in range(len(Q[row])):
        #~ update_q(row, column, column)
for i in range(2):
    update_q(0,4,4)
    update_q(1,3,3)
    update_q(1,5,5)
    update_q(2,3,3)
    update_q(3,1,1)
    update_q(3,2,2)
    update_q(3,4,4)
    update_q(4,0,0)
    update_q(4,3,3)
    update_q(4,5,5)
    update_q(5,1,1)
    update_q(5,4,4)
    update_q(5,5,5)

print Q
