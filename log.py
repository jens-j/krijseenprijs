#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(1, 2048, 2048)
y = 10**(0.0001617 * x)

# x = np.linspace(1, 2048, 2048)
# y = (1 - np.log10((2048 - x)) / np.log10(2048)) * 2048

print(x)
print(y)

fig, ax = plt.subplots()
ax.semilogy(x, y)
plt.show()
