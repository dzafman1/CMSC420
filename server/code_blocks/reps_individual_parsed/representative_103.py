# tree_reg1 = DecisionTreeRegressor(random_state=42)
# tree_reg2 = DecisionTreeRegressor(random_state=42, min_samples_leaf=10)
# tree_reg1.fit(X, y)
# tree_reg2.fit(X, y)
# 
# x1 = np.linspace(0, 1, 500).reshape(-1, 1)
# y_pred1 = tree_reg1.predict(x1)
# y_pred2 = tree_reg2.predict(x1)
# 
# plt.figure(figsize=(11, 4))
# 
# plt.subplot(121)
# plt.plot(X, y, "b.")
# plt.plot(x1, y_pred1, "r.-", linewidth=2, label=r"$\hat{y}$")
# plt.axis([0, 1, -0.2, 1.1])
# plt.xlabel("$x_1$", fontsize=18)
# plt.ylabel("$y$", fontsize=18, rotation=0)
# plt.legend(loc="upper center", fontsize=18)
# plt.title("No restrictions", fontsize=14)
# 
# plt.subplot(122)
# plt.plot(X, y, "b.")
# plt.plot(x1, y_pred2, "r.-", linewidth=2, label=r"$\hat{y}$")
# plt.axis([0, 1, -0.2, 1.1])
# plt.xlabel("$x_1$", fontsize=18)
# plt.title("min_samples_leaf={}".format(tree_reg2.min_samples_leaf), fontsize=14)
# 
# save_fig("tree_regression_regularization_plot")
# plt.show()

def computeDecisionTreeRegressor(df):
  from sklearn.tree import DecisionTreeRegressor
  from pandas.api.types import is_numeric_dtype
  tree_reg1 = DecisionTreeRegressor(random_state=42)
  quantitativeColumns = [c for c in list(df) if is_numeric_dtype(df[c])]
  X = df[quantitativeColumns[:-1]]
  y = df[[quantitativeColumns[-1]]]

  tree_reg1.fit(X, y)
  y_pred1 = tree_reg1.predict(X)
  res = X.copy()
  res["Expected-"+quantitativeColumns[-1]] = y
  res["Predicted-"+quantitativeColumns[-1]] = y_pred1
  return res

