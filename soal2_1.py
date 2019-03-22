import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

# Read a file
df = pd.read_csv('data.csv')

# Read a columns
# print(df.columns)

# add criteria as a reference
# print(df[df['Age'] <= 25])
# print(df[df['Overall'] >= 80])
# print(df[df['Potential'] >= 80])

# another reference for plotting and/or machine learning
# print(df[df['Age'] <= 25][df['Overall'] >= 80][df['Potential'] >= 80])

plt.figure('soal2_1_plot',figsize = (40,70))
plot1 = plt.subplot(121)
plt.scatter(
    df['Age'],
    df['Overall'],
    marker = 'o',
    color = 'r'
)
plt.scatter(
    df['Age'][df['Age'] <= 25][df['Overall'] >= 80],
    df['Overall'][df['Age'] <= 25][df['Overall'] >= 80],
    marker = 'o',
    color = 'g'
)
plot1.set_xlabel('Age')
plot1.set_ylabel('Overall')
plot1.set_title(' Age vs Overall')
plot1.legend(['Non-Target','Target'])
plot1.grid()
plot2 = plt.subplot(122)
plt.scatter(
    df['Age'],
    df['Potential'],
    marker = 'o',
    color = 'r'
)
plt.scatter(
    df['Age'][df['Age'] <= 25][df['Potential'] >= 80],
    df['Potential'][df['Age'] <= 25][df['Potential'] >= 80],
    marker = 'o',
    color = 'g'
)
plot2.set_xlabel('Age')
plot2.set_ylabel('Potential')
plot2.set_title('Age vs Potential')
plot2.legend(['Non-Target','Target'])
plot2.grid()
plt.show()