import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib.cbook

mean_placements = np.array([[2.52, 2.38, 2.02, 1.82, 2.08, 2.06, 1.98, 1.82, 1.88, 1.88],
 [2.22, 2.24, 2.26, 2.16, 2.04, 2.06, 1.72, 2.24, 1.8, 2.18,],
 [2.52, 2.38, 2.1,  2.18, 2.16, 1.94, 2.28, 2.04, 2.34, 1.92],
 [2.12, 2.44, 2.04, 2.36, 2.62, 2.64, 1.98, 2.1,  2.18, 1.6 ],
 [2.74, 2.2,  2.26, 2.8 , 2.22, 2.46, 2.32, 1.96, 2.1,  3.56],
 [2.26, 2.08, 1.96, 2.12, 2.56, 2.16, 2.22, 2.48, 2.78, 3.52],
 [2.48, 2.2,  3.14, 2.06, 1.98, 2.22, 2.08, 2.48, 2.6,  3.28],
 [2.06, 2.24, 3.36, 3.54, 2.64, 2.28, 2.48, 2.62, 2.46, 3.86],
 [3.48, 2.64, 3.38, 2.22, 3.52, 3.68, 2.62, 3.08, 3.7,  3.7 ],
 [3.46, 3.68, 2.48, 2.62, 2.38, 2.28, 2.62, 3.88, 3.88, 4.  ]])

lower_bound_gamma = 0
lower_bound_alpha = 0
upper_bound_gamma = 1
upper_bound_alpha = 1

step_number_alpha = 10
step_number_gamma = 10

step_size_alpha = (upper_bound_alpha - lower_bound_alpha)/step_number_alpha
step_size_gamma = (upper_bound_gamma - lower_bound_gamma)/step_number_gamma

## plot the winrate surface
x = np.linspace(lower_bound_alpha + step_size_alpha, upper_bound_alpha, step_number_alpha)
y = np.linspace(lower_bound_gamma + step_size_gamma, upper_bound_gamma, step_number_gamma)

X, Y = np.meshgrid(x, y)


fig = plt.figure(figsize=(8, 15))
ax = plt.axes(projection="3d")

plt.rc('lines', linewidth=2)
plt.rcParams.update({'font.size': 15})

ax.plot_wireframe(X, Y, mean_placements, color="navy")
ax.set_xlabel(r'$\gamma$')
ax.set_ylabel(r'$\alpha$')
ax.set_zlabel(r'average placement')

plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1])
plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1])

plt.xlim((0, 1))
plt.ylim((0, 1))

plt.savefig("optimization_landscape_qtable_1.png")
plt.show()

mean_placements = np.array([[2.7,  3.26, 2.75, 2.59, 2.4,  3.33, 3.29],
 [2.76, 2.44, 2.67, 3.08, 2.73, 3.52, 3.35],
 [2.84, 2.77, 2.66, 3.1,  2.58, 3.18, 3.84],
 [3.08, 2.55, 2.81, 2.73, 3.27, 2.76, 3.77],
 [2.59, 2.49, 2.49, 2.41, 2.95, 3.86, 3.94],
 [3.39, 2.83, 2.46, 2.44, 2.82, 3.1,  2.9 ],
 [3.35, 2.54, 3.57, 3.46, 2.74, 3.61, 3.9 ]])

step_number_alpha = 7
step_number_gamma = 7
lower_bound_alpha = 0.5
upper_bound_alpha = 0.9
lower_bound_gamma = 0.5
upper_bound_gamma = 1

step_size_alpha = (upper_bound_alpha - lower_bound_alpha)/step_number_alpha
step_size_gamma = (upper_bound_gamma - lower_bound_gamma)/step_number_gamma

## plot the winrate surface
x = np.linspace(lower_bound_alpha + step_size_alpha, upper_bound_alpha, step_number_alpha)
y = np.linspace(lower_bound_gamma + step_size_gamma, upper_bound_gamma, step_number_gamma)

X, Y = np.meshgrid(x, y)


fig = plt.figure(figsize=(8, 15))
ax = plt.axes(projection="3d")

plt.rc('lines', linewidth=2)
plt.rcParams.update({'font.size': 15})

ax.plot_wireframe(X, Y, mean_placements, color="navy")
ax.set_xlabel(r'$\gamma$')
ax.set_ylabel(r'$\alpha$')
ax.set_zlabel(r'average placement')


plt.xlim((lower_bound_gamma, upper_bound_gamma))
plt.ylim((lower_bound_alpha, upper_bound_alpha))

plt.savefig("second_plot.png")
plt.show()

mean_placements = np.array([[2.52, 2.38, 2.02, 1.82, 2.08, 2.06, 1.98, 1.82, 1.88, 1.88],
 [2.22, 2.24, 2.26, 2.16, 2.04, 2.06, 1.72, 2.24, 1.8, 2.18,],
 [2.52, 2.38, 2.1,  2.18, 2.16, 1.94, 2.28, 2.04, 2.34, 1.92],
 [2.12, 2.44, 2.04, 2.36, 2.62, 2.64, 1.98, 2.1,  2.18, 1.6 ],
 [2.74, 2.2,  2.26, 2.8 , 2.22, 2.46, 2.32, 1.96, 2.1,  3.56],
 [2.26, 2.08, 1.96, 2.12, 2.56, 2.16, 2.22, 2.48, 2.78, 3.52],
 [2.48, 2.2,  3.14, 2.06, 1.98, 2.22, 2.08, 2.48, 2.6,  3.28],
 [2.06, 2.24, 3.36, 3.54, 2.64, 2.28, 2.48, 2.62, 2.46, 3.86],
 [3.48, 2.64, 3.38, 2.22, 3.52, 3.68, 2.62, 3.08, 3.7,  3.7 ],
 [3.46, 3.68, 2.48, 2.62, 2.38, 2.28, 2.62, 3.88, 3.88, 4.  ]])

lower_bound_gamma = 0
lower_bound_alpha = 0
upper_bound_gamma = 1
upper_bound_alpha = 1

step_number_alpha = 10
step_number_gamma = 10

step_size_alpha = (upper_bound_alpha - lower_bound_alpha)/step_number_alpha
step_size_gamma = (upper_bound_gamma - lower_bound_gamma)/step_number_gamma

## plot the winrate surface
x = np.linspace(lower_bound_alpha + step_size_alpha, upper_bound_alpha, step_number_alpha)
y = np.linspace(lower_bound_gamma + step_size_gamma, upper_bound_gamma, step_number_gamma)

X, Y = np.meshgrid(x, y)


fig = plt.figure(figsize=(8, 15))
ax = plt.axes(projection="3d")

plt.rc('lines', linewidth=2)
plt.rcParams.update({'font.size': 15})

ax.plot_wireframe(X, Y, mean_placements, color="navy")
ax.set_xlabel(r'$\gamma$')
ax.set_ylabel(r'$\alpha$')
ax.set_zlabel(r'average placement')

plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1])
plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1])

plt.xlim((0, 1))
plt.ylim((0, 1))

plt.savefig("first_plot.png")
plt.show()
