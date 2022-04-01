import pickle
from matplotlib import pyplot as plt

figx = pickle.load(open('optimization_landscape_new.fig.pickle', 'rb'))

#plt.xlabel(r'$\gamma$')
#plt.ylabel(r'$\alpha$')

#plt.xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
#plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

#with open("optimization_landscape_new.fig.pickle", "wb") as file:
#    pickle.dump(figx, file)

plt.show()
