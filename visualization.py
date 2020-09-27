import function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

'''
generate data from functions:
s_f, b_f, sb_f, coc_f, d_f, h_b, i_b, m_p, c_m, ds_f, r_f, b_m, m_t
'''
_,_,_,x_r,y_r,z_r = function.m_t_para()
u_x = np.linspace(x_r[0], x_r[1], 200)
u_y = np.linspace(y_r[0], y_r[1], 200)
X, Y = np.meshgrid(u_x, u_y)
Z = function.m_t(X,Y)

'''
draw the 3d and contour pictures.
'''
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3, cmap=cm.winter_r)
# 绘制等高线
cset = ax.contour(X, Y, Z, zdir='z', offset=z_r[0], levels = 30, cmap=cm.coolwarm)
# cset = ax.contour(X, Y, Z, zdir='x', offset=x_r[0], cmap=cm.coolwarm)
# cset = ax.contour(X, Y, Z, zdir='y', offset=y_r[0], cmap=cm.coolwarm)

ax.set_xlabel('X')
ax.set_xlim(x_r[0], x_r[1])
ax.set_ylabel('Y')
ax.set_ylim(y_r[0], y_r[1])
ax.set_zlabel('Z')
ax.set_zlim(z_r[0], z_r[1])

plt.show()
