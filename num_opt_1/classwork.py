import pandas as pd
import numpy as np

#use pandas to read data
df = pd.read_csv("real_estate_dataset.csv")

#get number of samples and features from data
n_samples, n_features = df.shape

#get column names
columns = df.columns
print(columns)

#save column names to acess later as a text file
np.savetxt("columns_names.txt", columns, fmt = "%s")

#use Square_Feet, Garage_Size, Location_Score, Distance_to_Center, Year_Built as features
X = df[["Square_Feet", "Garage_Size", "Location_Score", "Distance_to_Center", "Year_Built"]]

#use price as the target
y = df["Price"].values

#get the numbers of samples and features
n_samples, n_features = X.shape

#build a linear model to predict price from the five features in X
#make an array of coefs of the size of n_features+1. Initialize to 1

coefs = np.ones(n_features+1)

#predict the price for each sample in X
predictions_bydefn = X@coefs[1:] + coefs[0]

#append a column of 1s to X
X = np.hstack((np.ones((n_samples,1)),X))

#predict the price for each sample in X
predictions = X@coefs

#see if all entries in predictions_bydefn and predictions are the same
is_same = np.allclose(predictions_bydefn, predictions)

print(is_same)

#calculate the error using predictions and y
errors = y - predictions
#calculate the relative error
rel_errors = errors/y
#calculate the mean of square of errors using a loop
loss_loop = 0
for i in range(n_samples):
    loss_loop = loss_loop + errors[i]**2

loss_loop = loss_loop/n_samples

#calculate the mean of square of errors using matrix ops
loss_matrix=np.transpose(errors)@errors/n_samples

#calculate the two methods to calculate mean of square of errors 
is_diff = np.allclose(loss_loop, loss_matrix)
print(f"are the loss by direct and matrix method same? {is_diff}/n")

#print the size of error and it's L2 norm
print(f"size of error is {errors.shape}")
print(f"L2 norm of error is {np.linalg.norm(errors)}")
print(f"L2 norm of error is {np.linalg.norm(rel_errors)}")

#writing the loss matrix
loss_matrix= (y-X@coefs).T@(y-X@coefs)/n_samples

#calculate the gradient of the loss function with respect to coefs
grad = -2*X.T@(y-X@coefs)/n_samples

#set the gradient equal to zero to get normal equation
coefs= np.linalg.inv(X.T@X)@X.T@y

#save coefs in file for viewing in text form
np.savetxt("coefs.csv", coefs, delimiter = ",")

#calcuate the predictions using optimal coefs
predictions_model = X@coefs
#calculate the errors using the optimal coefs
errors_model = y - predictions_model
#print the L2 norm of the error model
print(f"L2 norm of errors_model is {np.linalg.norm(errors_model)}")

#print the L2 norm of relative error model
rel_errors_model = errors_model/y

#use all the features in dataset to build a linear model to predict price
X = df.drop("Price", axis = 1).values
y = df["Price"].values
# get the number of samples and features
n_samples, n_features = X.shape

#solve the linear model using normal equations
x=np.hstack((np.ones((n_samples,1)),X))

#save coefficients to a file named coefs_all.csv
np.savetxt("coefs_all.csv", coefs, delimiter = ",")

#calcualte the rank of X.T@X
rank_XTX = np.linalg.matrix_rank(X.T@X)

#solve matrix equation using decomposition
#QR decomposition
Q,R = np.linalg.qr(X)

#write R to a file named R.csv
np.savetxt("R.csv", R, delimiter = ",")

#R.coefs = b

sol = Q.T@Q
np.savetxt("sol.csv", sol, delimiter = ",")
b=Q.T@y
coefs_qr = np.linalg.inv(R)@b  

#loop to solve R*coefs=b using back substitution
coefs_qr_loop = np.zeros(n_features+1)

for i in range(n_features, -1, -1):
    coefs_qr_loop[i] = b[i]
    for j in range(i+1, n_features+1):
        coefs_qr_loop[i] = coefs_qr_loop[i] - R[i,j]*coefs_qr_loop[j]
    coefs_qr_loop[i] = coefs_qr_loop[i]/R[i,i]

#save coefs_qr_loop to a file named coefs_qr_loop.csv
np.savetxt("coefs_qr_loop.csv", coefs_qr_loop, delimiter = ",")

#do eigen value decomposition of square matrix
#A=VDV^T
A = np.random.rand(4, 4)
A = A.T @ A
eigen_values, eigen_vectors = np.linalg.eig(A)

# Saving eigenvalues (as diagonal matrix) and eigenvectors
np.savetxt("eigen_values.csv", np.diag(eigen_values), delimiter=",")
np.savetxt("eigen_vectors.csv", eigen_vectors, delimiter=",")

# Singular Value Decomposition (SVD) and commit to repo
#  A = UΣV^T
U, S, Vt = np.linalg.svd(A)

# Saving the SVD components
np.savetxt("U.csv", U, delimiter=",")
np.savetxt("singular_values.csv", np.diag(S), delimiter=",")  
np.savetxt("Vt.csv", Vt, delimiter=",")


# Save the original matrix A
np.savetxt("A.csv", A, delimiter=",")









