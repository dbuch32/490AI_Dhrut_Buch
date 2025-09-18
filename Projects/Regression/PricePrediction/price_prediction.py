import pandas as pd
from sklearn.linear_model import LinearRegression

# Create a small dataset
data = {
"Size": [8, 12, 16, 20, 8, 12, 16, 20],
"Brand": ["A", "A", "A", "A", "B", "B", "B", "B"],
"Category": ["Soda", "Soda", "Soda", "Soda", "Juice", "Juice", "Juice", "Juice"],
"Price": [1.49, 1.99, 2.49, 2.89, 1.79, 2.19, 2.59, 2.99]
}

df = pd.DataFrame(data)
print(df)

# One-hot encode categorical features (drop_first avoids perfect multicollinearity)
df_encoded = pd.get_dummies(df, columns=["Brand", "Category"], drop_first=True)
print(df_encoded)

# Train linear regression
X = df_encoded.drop("Price", axis=1)
y = df_encoded["Price"]

model = LinearRegression()
model.fit(X, y)

print("Coefficients:", dict(zip(X.columns, model.coef_)))
print("Intercept:", model.intercept_)
print("R^2 (in-sample):", model.score(X, y))

# Predict price for: 16 oz, Brand B, Category=Juice
sample = pd.DataFrame({
"Size": [16],
"Brand_B": [1],
"Category_Juice": [1]
})

# Ensure column order matches training design matrix
sample = sample.reindex(columns=X.columns, fill_value=0)

pred = model.predict(sample)[0]
print("Predicted price:", round(pred, 2))