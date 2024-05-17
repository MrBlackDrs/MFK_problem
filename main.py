import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
from time import time


t1 = time()
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
# print(train)
train['mode'] = 'train'
test['mode'] = 'test'
df = pd.concat([train, test])
df = df.sort_values(['store_id', 'product_id', 'date']).reset_index(drop=True)
df.date = pd.to_datetime(df.date)
df['weekday'] = df.date.dt.weekday
# df.info()
# print(df.describe())
group = df.groupby(['store_id', 'product_id'])
for i in range(7, 21):
    df[f'lag_day_{i}'] = group['sales'].shift(i)


def metric_mape(y_valid: list, val_pred: list) -> float:
    """
    Calculate MAPE metric

    :param y_valid: list, real values
    :param val_pred: list, predicted values
    :return: float, MAPE metric
    """
    y_valid = np.array(y_valid).reshape(-1)
    val_pred = np.array(val_pred)
    return np.sum(np.abs(y_valid - val_pred)) / np.sum(y_valid) * 100


def calculate_metrics(df: pd.DataFrame) -> (float, float, float,
                                            float, float, float):
    """
    Calculate metrics

    :param df: pd.DataFrame, result dataframe
    :return:
    """
    y_true = df.prediction_y
    y_pr = df.prediction_x
    # y_naive = df.lag_day_7

    mae = mean_absolute_error(y_true, y_pr)
    mse = mean_squared_error(y_true, y_pr)
    # mae_naive = mean_absolute_error(y_true, y_naive)

    mape = metric_mape(y_true, y_pr)

    # mase = mae / mae_naive

    print(f"MAE: {mae:.2f}\n"
          f"MSE: {mse:.2f}\n"
          f"MAPE: {mape:.2f} %\n")


cat_features = ['store_id', 'category_id', 'product_id', 'weekday', 'city_name', 'weather_desc']
# числовые фичи
drop_features = ['id', 'date', 'sales', 'mode']
num_features = list(set(df.columns) - set(drop_features) - set(cat_features))
# Разбиваем датасет на train и test
train_df = df[(df['mode'] == 'train')].copy()
test_df = df[df['mode'] == 'test'].copy()

y_train = train_df.sales
y_test = test_df.sales

# категориальные фичи
X_train_cat = train_df[cat_features].values
X_test_cat = test_df[cat_features].values

# скалируем числовые фичи
X_train_num = train_df[num_features].copy()
X_test_num = test_df[num_features].copy()
scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train_num)
X_test_num = scaler.transform(X_test_num)

# итоговые датасеты для обучения
X_train = pd.DataFrame(np.hstack((X_train_num, X_train_cat)), columns=num_features+cat_features)
X_test = pd.DataFrame(np.hstack((X_test_num, X_test_cat)), columns=num_features+cat_features)


c_model = CatBoostRegressor(loss_function='MAE', random_state=42, silent=True, num_boost_round=300, max_depth=8)
c_model.fit(X_train, y_train, cat_features=cat_features, use_best_model=True)

pr = c_model.predict(X_test)
test_df['forecast'] = pr
prediction = test_df[['id', 'forecast']]
prediction.columns = ['id', 'prediction']
final = prediction[['id', 'prediction']].copy()
final['prediction'] = ["{:.1f}".format(i) for i in final['prediction']]
final.to_csv('prediction.csv', index=False, encoding='utf-8')


pr = c_model.predict(X_train)
w = pd.DataFrame({'prediction_x': pr, 'prediction_y': train_df['sales']})
calculate_metrics(w)

t2 = time()
print(t2 - t1, " sec or ", (t2-t1)/60, " min")

