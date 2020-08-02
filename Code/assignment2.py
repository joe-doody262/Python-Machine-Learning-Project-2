import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb


dataset = pd.read_csv('CMP3751M_CMP9772M_ML_Assignment 2-dataset-nuclear_plants_final.csv').values #reads data values of the dataset file
dataset_matrix = pd.read_csv('CMP3751M_CMP9772M_ML_Assignment 2-dataset-nuclear_plants_final.csv') #Reads the data but is used to make boxplots

#Each column in the dataset
power_range_sensor1 = dataset[:,1] #Power_range_sensor_1 column in the dataset
power_range_sensor2 = dataset[:,2] #Power_range_sensor_2 column in the dataset
power_range_sensor3 = dataset[:,3] #Power_range_sensor_3 column in the dataset
power_range_sensor4 = dataset[:,4] #Power_range_sensor_4 column in the dataset
pressure_sensor1 = dataset[:,5] #Pressure_sensor_1 column in the dataset
pressure_sensor2 = dataset[:,6] #Pressure_sensor_2 column in the dataset
pressure_sensor3 = dataset[:,7] #Pressure_sensor_3 column in the dataset
pressure_sensor4 = dataset[:,8] #Pressure_sensor_4 column in the dataset
vibration_sensor1 = dataset[:,9] #Vibration_sensor_1 column in the dataset
vibration_sensor2 = dataset[:,10] #Vibration_sensor_2 column in the dataset
vibration_sensor3 = dataset[:,11] #Vibration_sensor_3 column in the dataset
vibration_sensor4 = dataset[:,12] #Vibration_sensor_4 column in the dataset

#calculate the minimum
power_range_sensor1_min = np.amin(power_range_sensor1) #Calculates the minimum of Power_range_sensor_1
power_range_sensor2_min = np.amin(power_range_sensor2) #Calculates the minimum of Power_range_sensor_2
power_range_sensor3_min = np.amin(power_range_sensor3) #Calculates the minimum of Power_range_sensor_3
power_range_sensor4_min = np.amin(power_range_sensor4) #Calculates the minimum of Power_range_sensor_4
pressure_sensor1_min = np.amin(pressure_sensor1) #calculates the minimum of Pressure_sensor_1
pressure_sensor2_min = np.amin(pressure_sensor2) #calculates the minimum of Pressure_sensor_2
pressure_sensor3_min = np.amin(pressure_sensor3) #calculates the minimum of Pressure_sensor_3
pressure_sensor4_min = np.amin(pressure_sensor4) #calculates the minimum of Pressure_sensor_4
vibration_sensor1_min = np.amin(vibration_sensor1) #calculates the minimum of Vibration_sensor_1
vibration_sensor2_min = np.amin(vibration_sensor2) #calculates the minimum of Vibration_sensor_2
vibration_sensor3_min = np.amin(vibration_sensor3) #calculates the minimum of Vibration_sensor_3
vibration_sensor4_min = np.amin(vibration_sensor4) #calculates the minimum of Vibration_sensor_4                             
print('Power_range_sensor_1 minimum: ' + str(power_range_sensor1_min)) #outputs the minimum of Power_range_sensor_1
print('Power_range_sensor_2 minimum: ' + str(power_range_sensor2_min)) #outputs the minimum of Power_range_sensor_2
print('Power_range_sensor_3 minimum: ' + str(power_range_sensor3_min)) #outputs the minimum of Power_range_sensor_3
print('Power_range_sensor_4 minimum: ' + str(power_range_sensor4_min)) #outputs the minimum of Power_range_sensor_4
print('Pressure_sensor_1 minimum: ' + str(pressure_sensor1_min)) #Outputs the minimum of Pressure_sensor_1
print('Pressure_sensor_2 minimum: ' + str(pressure_sensor2_min)) #Outputs the minimum of Pressure_sensor_2
print('Pressure_sensor_3 minimum: ' + str(pressure_sensor3_min)) #Outputs the minimum of Pressure_sensor_3
print('Pressure_sensor_4 minimum: ' + str(pressure_sensor4_min)) #Outputs the minimum of Pressure_sensor_4
print('Vibration_sensor_1 minimum: ' + str(vibration_sensor1_min)) #Outputs the minimum of Vibration_sensor_1
print('Vibration_sensor_2 minimum: ' + str(vibration_sensor2_min)) #Outputs the minimum of Vibration_sensor_2
print('Vibration_sensor_3 minimum: ' + str(vibration_sensor3_min)) #Outputs the minimum of Vibration_sensor_3
print('Vibration_sensor_4 minimum: ' + str(vibration_sensor4_min)) #Outputs the minimum of Vibration_sensor_4

#calculate the maximum
power_range_sensor1_max = np.amax(power_range_sensor1) #Calculates the maximum of Power_range_sensor_1
power_range_sensor2_max = np.amax(power_range_sensor2) #Calculates the maximum of Power_range_sensor_2
power_range_sensor3_max = np.amax(power_range_sensor3) #Calculates the maximum of Power_range_sensor_3
power_range_sensor4_max = np.amax(power_range_sensor4) #Calculates the maximum of Power_range_sensor_4
pressure_sensor1_max = np.amax(pressure_sensor1) #calculates the maximum of Pressure_sensor_1
pressure_sensor2_max = np.amax(pressure_sensor2) #calculates the maximum of Pressure_sensor_2
pressure_sensor3_max = np.amax(pressure_sensor3) #calculates the maximum of Pressure_sensor_3
pressure_sensor4_max = np.amax(pressure_sensor4) #calculates the maximum of Pressure_sensor_4
vibration_sensor1_max = np.amax(vibration_sensor1) #calculates the maximum of Vibration_sensor_1
vibration_sensor2_max = np.amax(vibration_sensor2) #calculates the maximum of Vibration_sensor_2
vibration_sensor3_max = np.amax(vibration_sensor3) #calculates the maximum of Vibration_sensor_3
vibration_sensor4_max = np.amax(vibration_sensor4) #calculates the maximum of Vibration_sensor_4                             
print('Power_range_sensor_1 maximum: ' + str(power_range_sensor1_max)) #outputs the maximum of Power_range_sensor_1
print('Power_range_sensor_2 maximum: ' + str(power_range_sensor2_max)) #outputs the maximum of Power_range_sensor_2
print('Power_range_sensor_3 maximum: ' + str(power_range_sensor3_max)) #outputs the maximum of Power_range_sensor_3
print('Power_range_sensor_4 maximum: ' + str(power_range_sensor4_max)) #outputs the maximum of Power_range_sensor_4
print('Pressure_sensor_1 maximum: ' + str(pressure_sensor1_max)) #Outputs the maximum of Pressure_sensor_1
print('Pressure_sensor_2 maximum: ' + str(pressure_sensor2_max)) #Outputs the maximum of Pressure_sensor_2
print('Pressure_sensor_3 maximum: ' + str(pressure_sensor3_max)) #Outputs the maximum of Pressure_sensor_3
print('Pressure_sensor_4 maximum: ' + str(pressure_sensor4_max)) #Outputs the maximum of Pressure_sensor_4
print('Vibration_sensor_1 maximum: ' + str(vibration_sensor1_max)) #Outputs the maximum of Vibration_sensor_1
print('Vibration_sensor_2 maximum: ' + str(vibration_sensor2_max)) #Outputs the maximum of Vibration_sensor_2
print('Vibration_sensor_3 maximum: ' + str(vibration_sensor3_max)) #Outputs the maximum of Vibration_sensor_3
print('Vibration_sensor_4 maximum: ' + str(vibration_sensor4_max)) #Outputs the maximum of Vibration_sensor_4

#calculate the mean
power_range_sensor1_mean = np.mean(power_range_sensor1) #calculates the mean of Power_range_sensor_1
power_range_sensor2_mean = np.mean(power_range_sensor2) #calculates the mean of Power_range_sensor_2
power_range_sensor3_mean = np.mean(power_range_sensor3) #calculates the mean of Power_range_sensor_3
power_range_sensor4_mean = np.mean(power_range_sensor4) #calculates the mean of Power_range_sensor_4
pressure_sensor1_mean = np.mean(pressure_sensor1) # calculates the mean of Pressure_sensor_1
pressure_sensor2_mean = np.mean(pressure_sensor2) # calculates the mean of Pressure_sensor_2
pressure_sensor3_mean = np.mean(pressure_sensor3) # calculates the mean of Pressure_sensor_3
pressure_sensor4_mean = np.mean(pressure_sensor4) # calculates the mean of Pressure_sensor_4
vibration_sensor1_mean = np.mean(vibration_sensor1) #calculates the mean of Vibration_sensor_1
vibration_sensor2_mean = np.mean(vibration_sensor2) #calculates the mean of Vibration_sensor_1
vibration_sensor3_mean = np.mean(vibration_sensor3) #calculates the mean of Vibration_sensor_1
vibration_sensor4_mean = np.mean(vibration_sensor4) #calculates the mean of Vibration_sensor_1
print('Power_range_sensor_1 mean: ' + str(power_range_sensor1_mean)) #outputs the mean of Power_range_sensor_1
print('Power_range_sensor_2 mean: ' + str(power_range_sensor2_mean)) #outputs the mean of Power_range_sensor_2
print('Power_range_sensor_3 mean: ' + str(power_range_sensor3_mean)) #outputs the mean of Power_range_sensor_3
print('Power_range_sensor_4 mean: ' + str(power_range_sensor4_mean)) #outputs the mean of Power_range_sensor_4
print('Pressure_sensor_1 mean: ' + str(pressure_sensor1_mean)) #outputs the mean of Pressure_sensor_1
print('Pressure_sensor_2 mean: ' + str(pressure_sensor2_mean)) #outputs the mean of Pressure_sensor_2
print('Pressure_sensor_3 mean: ' + str(pressure_sensor3_mean)) #outputs the mean of Pressure_sensor_3
print('Pressure_sensor_4 mean: ' + str(pressure_sensor4_mean)) #outputs the mean of Pressure_sensor_4
print('Vibration_sensor_1 mean: ' + str(vibration_sensor1_mean)) #outputs the mean of Vibration_sensor_1
print('Vibration_sensor_2 mean: ' + str(vibration_sensor2_mean)) #outputs the mean of Vibration_sensor_2
print('Vibration_sensor_3 mean: ' + str(vibration_sensor3_mean)) #outputs the mean of Vibration_sensor_3
print('Vibration_sensor_4 mean: ' + str(vibration_sensor4_mean)) #outputs the mean of Vibration_sensor_4

#calculate the variance
power_range_sensor1_var = np.var(power_range_sensor1) #calculates the variance of Power_range_sensor_1
power_range_sensor2_var = np.var(power_range_sensor2) #calculates the variance of Power_range_sensor_2
power_range_sensor3_var = np.var(power_range_sensor3) #calculates the variance of Power_range_sensor_3
power_range_sensor4_var = np.var(power_range_sensor4) #calculates the variance of Power_range_sensor_4
pressure_sensor1_var = np.var(pressure_sensor1) # calculates the variance of Pressure_sensor_1
pressure_sensor2_var = np.var(pressure_sensor2) # calculates the variance of Pressure_sensor_2
pressure_sensor3_var = np.var(pressure_sensor3) # calculates the variance of Pressure_sensor_3
pressure_sensor4_var = np.var(pressure_sensor4) # calculates the variance of Pressure_sensor_4
vibration_sensor1_var = np.var(vibration_sensor1) #calculates the variance of Vibration_sensor_1
vibration_sensor2_var = np.var(vibration_sensor2) #calculates the variance of Vibration_sensor_1
vibration_sensor3_var = np.var(vibration_sensor3) #calculates the variance of Vibration_sensor_1
vibration_sensor4_var = np.var(vibration_sensor4) #calculates the variance of Vibration_sensor_1
print('Power_range_sensor_1 variance: ' + str(power_range_sensor1_var)) #outputs the variance of Power_range_sensor_1
print('Power_range_sensor_2 variance: ' + str(power_range_sensor2_var)) #outputs the variance of Power_range_sensor_2
print('Power_range_sensor_3 variance: ' + str(power_range_sensor3_var)) #outputs the variance of Power_range_sensor_3
print('Power_range_sensor_4 variance: ' + str(power_range_sensor4_var)) #outputs the variance of Power_range_sensor_4
print('Pressure_sensor_1 variance: ' + str(pressure_sensor1_var)) #outputs the variance of Pressure_sensor_1
print('Pressure_sensor_2 variance: ' + str(pressure_sensor2_var)) #outputs the variance of Pressure_sensor_2
print('Pressure_sensor_3 variance: ' + str(pressure_sensor3_var)) #outputs the variance of Pressure_sensor_3
print('Pressure_sensor_4 variance: ' + str(pressure_sensor4_var)) #outputs the variance of Pressure_sensor_4
print('Vibration_sensor_1 variance: ' + str(vibration_sensor1_var)) #outputs the variance of Vibration_sensor_1
print('Vibration_sensor_2 variance: ' + str(vibration_sensor2_var)) #outputs the variance of Vibration_sensor_2
print('Vibration_sensor_3 variance: ' + str(vibration_sensor3_var)) #outputs the variance of Vibration_sensor_3
print('Vibration_sensor_4 variance: ' + str(vibration_sensor4_var)) #outputs the variance of Vibration_sensor_4

#create the boxplot for Vibration_sensor_1 for normal and abnormal data
sensor1 = dataset_matrix['Vibration_sensor_1'].as_matrix() #converst the dataset column to a matrix
status = dataset_matrix['Status'].as_matrix() #converts the ataset column to a matrix
sb.boxplot(status, sensor1) #Create the boxplot for the data
plt.show() #displays the boxplot

#sensor2 = dataset_matrix['Vibration_sensor_2'].as_matrix()
#Density plot
normal_dens = dataset_matrix[dataset_matrix['Status'] == 'Normal'].loc[:, ['Vibration_sensor_2']] #Gathers the normal data
abnormal_dens = dataset_matrix[dataset_matrix['Status'] == 'Abnormal'].loc[:, ['Vibration_sensor_2']] #gathers the abnormal data

df = pd.DataFrame({'Normal': normal_dens['Vibration_sensor_2'], 'Abnormal': abnormal_dens['Vibration_sensor_2']}) #Create a dataframe for the gathered data
df.plot.kde() #Plots the density plot
