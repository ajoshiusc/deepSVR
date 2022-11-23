from matplotlib import pyplot as plt
import numpy as np

a = np.load('/home/ajoshi/svr_reg_epoch_loss_values_on_the_fly.npz')

a.keys

plt.plot(a['epoch_loss_valid'])
#plt.plot(a['epoch_loss_values'][::10])
plt.show()