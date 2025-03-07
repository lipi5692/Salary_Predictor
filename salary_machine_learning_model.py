import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:, :-1].values  # Independent variable (Years of Experience)
Y = dataset.iloc[:, -1].values   # Dependent variable (Salary)

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train the Linear Regression model
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predict salaries for the test set
Y_pred = regressor.predict(X_test)

# Create a DataFrame for comparison
df = pd.DataFrame({
    "Year of Experience": X_test.flatten(),
    "Predicted Salary": Y_pred,
    "Real Salary": Y_test
})
print(df)

# Plot training set results
plt.scatter(X_train, Y_train, color='red', label='Actual Salary')
plt.plot(X_train, regressor.predict(X_train), color='blue', label='Regression Line')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()

# Plot test set results
plt.scatter(X_test, Y_test, color='red', label='Actual Salary')
plt.plot(X_train, regressor.predict(X_train), color='blue', label='Regression Line')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()
