import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn.model_selection import train_test_split
from keras import models, layers, initializers
from keras.utils.np_utils import to_categorical
from datetime import datetime, timedelta
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from keras.optimizers import RMSprop
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


f1 = pd.read_csv('./data_utf.csv', encoding='utf-8') #気象データ
f2 = pd.read_csv('./patients.csv', encoding='utf-8') #感染者数と検査数のデータ

columns = f1.columns.tolist()
res_pd = f1.drop([f1.columns[2], f1.columns[3], f1.columns[5], f1.columns[6], f1.columns[8], f1.columns[9], f1.columns[11], f1.columns[12], f1.columns[14], f1.columns[15]], axis=1)
res_pd.columns = res_pd.iloc[0, :].tolist()
res_pd = res_pd.drop([res_pd.index[0], res_pd.index[1], res_pd.index[2]]).reset_index(drop=True)
res_pd = res_pd[23:366].reset_index(drop=True)

res = pd.concat([res_pd, f2.iloc[:, 1:3]], axis=1)
columns = res.columns.values.tolist()
ohe_cols = [columns[4], columns[5]]
ce_ohe = ce.OneHotEncoder(cols=ohe_cols, handle_unknown='Nan')
a = ce_ohe.fit(res)
res = a.transform(res)

date_list = [datetime(2020, 1, 24) + timedelta(days=i) for i in range(len(res))]
res = pd.concat([pd.DataFrame({"日付": date_list}), res], axis=1)
res = res.drop("年月日", axis=1)
columns = res.columns.tolist()

res['日付'] = res["日付"].astype("int64").values.astype("float64").reshape(-1, 1) // 10**9
for i in range(1, 4):
    res[columns[i]] = res[columns[i]].astype('float64')

for i in range(4):
    res[columns[i]] = (res[columns[i]] - res[columns[i]].mean()) / res[columns[i]].std(ddof=0)

X = res.iloc[:, :-2]
Y = res.iloc[:, -2].values.reshape(-1, 1)

n_rnn = 14
n_sample = X.shape[0] - n_rnn
X_array = np.zeros((n_sample, n_rnn, 116))
y_array = np.zeros((n_sample, n_rnn, 1))

for i in range(n_sample):
    X_array[i] = X[i:i+n_rnn]
    y_array[i] = Y[i:i+n_rnn]

X_train = X_array[:264]
y_train = y_array[:264]
X_test = X_array[264:]
y_test = y_array[264:]

#モデルの構築
batch_size = 5  # バッチサイズ
n_in = 116  # 入力層のニューロン数
n_mid = 32  # 中間層のニューロン数
n_out = 1  # 出力層のニューロン数

model = Sequential()
model.add(layers.GRU(n_mid, return_sequences=True, input_shape=(None, n_in)))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1))
model.compile(loss="mse", optimizer=RMSprop())

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=batch_size, validation_split=0.1)

test_pre = model.predict(X_test)
train_pre = model.predict(X_train)

test = np.zeros([78, 2])
for i in range(test_pre.shape[0]):
    for j in range(14):
        test[i+j][0] += test_pre[i][j][0]
        test[i+j][1] += 1
test_ave = []
for i in range(len(test)):
    test_ave.append(test[i][0] / test[i][1])

train = np.zeros([265, 2])
    
for i in range(train_pre.shape[0]-12):
    for j in range(14):
        train[i+j][0] += train_pre[i][j][0]
        train[i+j][1] += 1
train_ave = []
for i in range(len(train)):
    train_ave.append(train[i][0] / train[i][1])

X_axis = res["日付"]
Y_axis = train_ave + test_ave
Y_true = res["新規感染者数"].values.tolist()

f = plt.figure(figsize=(10, 8))
plt.plot(X_axis[:265], train_ave, "--", label="predicted_train")
plt.plot(X_axis[265:], test_ave, "--", label="predicted_test")
plt.plot(X_axis[:265], Y_true[:265], label="true_train")
plt.plot(X_axis[265:], Y_true[265:], label="true_test")

plt.legend()
plt.savefig('./predicted_korona.jpg')
