import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def get_vertices(x, y, theta_rad, L, W):
    """
    x, y: 形心坐标
            rear          front
             D              A
             C              B
    """
    if isinstance(theta_rad, np.matrix):
        x = np.matrix(x).transpose()
        x = np.kron(x, np.ones_like(theta_rad[0, :]))
    elif isinstance(theta_rad, np.ndarray):
        x = np.matrix(x).transpose()
        y = np.matrix(y).transpose()
        theta_deg = np.matrix(theta_rad).transpose()
        x = np.kron(x, np.ones_like(theta_deg[0, :]))
    # A
    AE = ((W / 2) ** 2 + (L / 2) ** 2) ** 0.5
    angleAEF = np.arctan(W / 2 / (L / 2))
    xA = x + AE * np.cos(theta_rad + angleAEF)
    yA = y + AE * np.sin(theta_rad + angleAEF)
    # B
    BE = AE
    angleBEF = angleAEF
    xB = x + BE * np.cos(theta_rad - angleBEF)
    yB = y + BE * np.sin(theta_rad - angleBEF)
    # C
    xC = x - (xA - x)
    yC = y - (yA - y)
    # D
    xD = x - (xB - x)
    yD = y - (yB - y)
    return xA, yA, xB, yB, xC, yC, xD, yD


def add_rectangle(ax, xy, L, W, theta_rad, agent_type='EGO'):
    if agent_type == 'EGO':
        color = [0.15, 0.53, 0.79]
    elif agent_type == 'VEHICLE':
        color = [0.55, 0.33, 0.82]

    rect = patches.Rectangle(xy, L, W, angle=np.rad2deg(theta_rad))
    ax.add_patch(rect)

#%% performance comparison
import time
t1 = time.time()
for _ in range(1000):
    pass
t2 = time.time()
print('elapsed time:%s ms' % ((t2 - t1) * 1000))
t1 = time.time()
for _ in range(1000):
    pass
t2 = time.time()
print('elapsed time:%s ms' % ((t2 - t1) * 1000))