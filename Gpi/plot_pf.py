import pandas as pd
from matplotlib import pyplot as plt

columns = ["objective_1", "objective_2"]

df = pd.read_csv("GPI/2obj_norm_pen.csv", usecols=columns)

plt.plot(df.objective_1, df.objective_2, 'o')
plt.xlabel('Cost due to excess water level wrt flooding threshold upstream')
plt.ylabel('Deficit in water supply wrt demand')
plt.show()