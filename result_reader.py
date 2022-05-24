import numpy as np
import matplotlib.pyplot as plt

filename = "results.npy"

arr = np.load(filename)
#print(arr)
print(arr[-1])

scores = []
timesteps = []

for i in range(len(arr) - 100):
    scores.append(np.average(arr[i:i+100,0]))
    timesteps.append(arr[i + 100][3] / 3600)

plt.plot(timesteps,scores)
plt.ylabel('Average over last 100 games')
plt.xlabel('Wall Time (Hours)')
plt.show()
