import pandas as pd

#Le o arquivo
base = pd.read_csv('credit_data.csv')
#Apaga os dados faltantes
base = base.dropna()

#Mostra os outliers da idade 
import matplotlib.pyplot as plt
plt.boxplot(base.iloc[:,2], showfliers = True)
outliers_age = base[(base.age < -20)]

# Mostra os outliers do loan
plt.boxplot(base.iloc[:,3])
outliers_loan = base[(base.loan > 13400)]