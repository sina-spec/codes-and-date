import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as mse, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RepeatedKFold
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import joblib
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor as KNR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import GradientBoostingRegressor as GBR
import matplotlib

matplotlib.use('Agg')
plt.rcParams['font.family'] = 'Times New Roman'

data1 = pd.read_csv('data_001.csv', skiprows=1, header=None)
data2 = pd.read_csv('data_111.csv', skiprows=1, header=None)
combined_data = pd.concat([data1, data2], ignore_index=True)

dataset_label = pd.read_csv('dataset_label.csv')
y = dataset_label['label'].values

x = combined_data.iloc[:, 2:].values

ss = StandardScaler()
x = ss.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


gbr_params = {
     'n_estimators': 200,
     'learning_rate': 0.05,
     'max_depth': 6,
     'min_samples_leaf': 10,
     'max_features': 'sqrt',
 }

gbr_param_grid = {
    'n_estimators': [400, 500, 600],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [2, 3, 4],
    'min_samples_leaf': [10,15,20],
}

cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)


model_gbr = GBR(**gbr_params)
grid_search = GridSearchCV(estimator=model_gbr,
                           param_grid=gbr_param_grid,
                           scoring='neg_mean_squared_error',
                           cv=cv,
                           return_train_score=True,
                           verbose=2)
grid_search.fit(x_train, y_train)
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

joblib.dump(best_model, 'model_gbr.pkl')

y_train = y_train.reshape(-1, 1)
predictions = best_model.predict(x_train).reshape(-1, 1)
dottrain = np.hstack((y_train, predictions))

y_test = y_test.reshape(-1, 1)
predictions_t = best_model.predict(x_test).reshape(-1, 1)
dottest = np.hstack((y_test,predictions_t))

y_train_pred = best_model.predict(x_train)
rmse_train = np.sqrt(mse(y_train, y_train_pred))
r2_train = r2_score(y_train, y_train_pred)
y_test_pred = best_model.predict(x_test)
rmse_test = np.sqrt(mse(y_test, y_test_pred))
r2_test = r2_score(y_test, y_test_pred)

np.savetxt('dottrain.csv', dottrain, delimiter=',')
np.savetxt('dottest.csv', dottest, delimiter=',')

print("Best parameters found using GridSearchCV:", best_params)
print("RMSE_train:", rmse_train)
print("R2_train:", r2_train)
print("RMSE_test:", rmse_test)
print("R2_test:", r2_test)



with open("results_gbr.txt", "w") as f:
    f.write("Best parameters found using GridSearchCV:\n")
    f.write(f"{best_params}\n")
    f.write("\nPerformance Metrics:\n")
    f.write(f"RMSE_train: {rmse_train:.3f}\n")
    f.write(f"R2_train: {r2_train:.3f}\n")
    f.write(f"RMSE_test: {rmse_test:.3f}\n")
    f.write(f"R2_test: {r2_test:.3f}\n")
print("Results have been saved.")

plt.figure(figsize=(8, 7))
plt.rcParams.update({'font.size': 14})
plt.rcParams['font.family']='Times New Roman'
plt.scatter(y_train, best_model.predict(x_train), color='blue',label='train data')
plt.scatter(y_test, best_model.predict(x_test), color='red',label='test data')

plt.legend([f'Train set , RMSE = {rmse_train:.3f}, R2 = {r2_train:.3f}',
           f'Test set , RMSE = {rmse_test:.3f}, R2 = {r2_test:.3f}'],
           labelcolor=['blue', 'red'])

plt.plot([-1.5,1.5], [-1.5,1.5], 'k', alpha=0.5)
plt.plot([-1.5,1.5], [-1.3,1.7], 'k--', alpha=0.5)
plt.plot([-1.5,1.5], [-1.7,1.3], 'k--', alpha=0.5)

plt.tick_params(which='both',direction='in', left='True', bottom='True', width=1, pad=12)

plt.xlabel(r'$\Delta E_{\mathrm{OH*}}^{DFT} \, (\mathrm{eV})$', fontsize=12, fontfamily='Times New Roman')
plt.ylabel(r'$\Delta E_{\mathrm{OH*}}^{ML} \, (\mathrm{eV})$', fontsize=12, fontfamily='Times New Roman')
plt.xlim(-1.5,1.5)
plt.ylim(-1.5,1.5)
plt.box(on=True)


plt.savefig('PredictionError_gbr.png', dpi=300)
plt.show()
plt.clf()