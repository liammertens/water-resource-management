import pandas as pd
from matplotlib import pyplot as plt

columns = ["objective_1", "objective_2", "objective_3", "objective_4"]

df = pd.read_csv("GPI/wandb_export_2023-03-23T18_23_52.285+01_00.csv", usecols=columns)

plt.plot(df.objective_3, df.objective_4)
plt.show()