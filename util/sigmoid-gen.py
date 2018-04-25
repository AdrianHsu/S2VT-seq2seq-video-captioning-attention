import numpy as np
import matplotlib.pyplot as plt

num_epo = 100
x = np.arange(-2.0, 2.0, (4.0/num_epo))
y = 1 / (1 + np.e**x)
print(y)
plt.axis([0, 100, 0, 1])
plt.plot(y)
plt.title('inv-sigmoid')
plt.xlabel('epoch')
plt.ylabel('probs')
plt.savefig('sigmoid.png')
