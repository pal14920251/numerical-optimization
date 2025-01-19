import numpy as np
import pandas as pd


# use pandas to read the data
df = pd.read_csv("real_estate_dataset.csv")

# get number of samples and features
n_samples , n_features = df.shape

# get column names

columns = df.columns

#save the column names to access later as a text file

np.savetxt("columns_names.txt", columns, fmt = "%s")

# Use Square_Feet, Garage_Size, Location_Score, Distance_to_Center as features

X = df[["Square_Feet", "Garage_Size", "Location_Score", "Distance_to_Center"]]

# Use price as the target
y = df["Price"].values

# get number of samples and features
n_samples , n_features = X.shape

#Build a linear model to predict price from the four features in X
# make an array of coefs of the size of n_featres+1. InitiLalize to 1

coefs = np.ones(n_features+1)

# predict the price for each sample in X
predictions_bydefn = X@coefs[1:] + coefs[0]

#append a column of 1s to X
X = np.hstack(np.ones((n_samples,1)),X)

# predict the price for each sample in X
predictions = X@coefs

# see if all entries in predictions_bydefn and predictions are the same
is_same = np.allclose(predictions_bydefn , predictions)

print(is_same)
# calculate the error using predictions and y

errors = y - predictions
# calculate the relative error 
rel_errors = errors/y
#calculate the mean of square of errors using a loop

loss_loop = 0
for i in range(n_samples):
    loss_loop = loss_loop + errors[i]**2

loss_loop = loss_loop/n_samples

# calculate the mean of square of errors using matrix ops
loss_matrix = np.transpose(errors)@errors/n_samples

# calculate the two methds of calculating mean squared error
is_diff = np.allclose(loss_loop , loss_matrix)
print(f"Are the loss by direct and matrix same? {is_diff}\n")

# print the size of errors and it's L2 norm
print(f"Size of erros: {errors.shape}")
print(f"L2 norm of errors: {np.linalg.norm(errors)}")
print(f"L2 norm of errors: {np.linalg.norm(rel_errors)}")

# Writing the loss matrix

loss_matrix = (y-X@coefs).T@(y-X@coefs)/n_samples

# Calculate te he gradient of the loss with respect to the coefficients 
grad_matrix = -2/n_samples*X.T@(y-X@coefs)

# Set gradient equal to zero to get normal equations

coefs = np.lialg.inv(X.T @ X) @ X.T @ y

# save coefs to a file for viewing
np.savetxt("coefs.csv" , coefs, delimiter = ",")

# calculate yhe predictions using optimal coefficients
predictions_model = X@coefs
# calculate the errors using the optimal coefficients
errors_model = y - predictions_model
# print the L2 norm of the errors_model
print(f"L2 norm of errors_model: {np.linalg.norm(errors_model)}")
# print the L2 norm of the relative errors_model
rel_errors_model = errors_model/y

# Use all the features in the dataset to build a linear model to predict prices 
X = df.drop("Price" , axis = 1).values
y = df["Price"].values

# get the number of smaples and features inn X

n_samples , n_features = X.shape

# solve the linear model using the normal equations 
X = np.hstack(np.ones((n_samples,1)),X)

# save coefs to a file named coefs_all.csv
np.savetxt("coefs_all.csv" , coefs , delimiter = ",")

# calculate the rank of X.T @ X
rank_XTX = np.linalg.matrix_rank(X.T @ X)

# solve matrix equations using decomposition
# QR factorization

Q, R = np.linalg.qr(X)

# write R to a file named R.csv
np.savetxt("R.csv" , R , delimiter = ",")

# R*coefs = b

sol = Q.T @ Q
np.savetxt("sol.csv" , sol , delimiter = ",")

b = Q.T @ y
coeffs_qr = np.linalg.inv(R) @ b

# loop to solve R*coeffs = b using back substitution
coeffs_qr_loop = np.zeros(n_features+1)
for i in range(n_features , -1 , -1):
    coeffs_qr_loop[i] = b[i]
    for j in range(i+1 , n_features+1):
        coeffs_qr_loop[i] = coeffs_qr_loop[i] - R[i,j]*coeffs_qr_loop[j]
    
    coeffs_qr_loop[i] = coeffs_qr_loop[i]/R[i,i]

# save coeffs_qr_loop to a file named coeffs_qr_loop.csv
np.savetxt("coeffs_qr_loop.csv" , coeffs_qr_loop, delimiter = ",")

# Eigen decomposition of a quare matrix

#A = VDV^T

# Take SVD and commit to repo
