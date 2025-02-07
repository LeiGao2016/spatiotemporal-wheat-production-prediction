import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.layers import ConvLSTM2D, Flatten, Dense, Reshape, Dropout, Input, BatchNormalization
from tensorflow.keras.models import Sequential, Model, Sequential
from tensorflow.keras.optimizers import Adam

tf.config.list_physical_devices('GPU')

# 提取指定年份的数据
def extract_yearly_data(year):
    year_data = []
    for var in dynamic_vars:
        year_var = f"{var}_{year}"
        if year_var in dataset.columns:
            year_data.append(dataset[year_var].values.reshape(-1, 1))
    return np.hstack(year_data)

def extract_yearly_target_data(year):
    year_data = []
    year_var = f"Yield_Grid_{year}"
    year_data.append(dataset[year_var].values.reshape(-1, 1))
    return np.hstack(year_data)

# 提取静态变量数据
def extract_static_data():
    static_data = []
    for var in static_vars:
        static_data.append(dataset[var].values.reshape(-1, 1))
    return np.hstack(static_data)

# 创建输入输出张量
def create_single_input_output_tensor(start_year):
    static_data = extract_static_data()
    num_regions = static_data.shape[0]
    num_batches = num_regions // 313

    input_data = []
    output_data = []

    for batch_idx in range(num_batches):
        batch_start = batch_idx*313
        batch_end = batch_start+313

        input_batch = []
        output_batch = []

        for year in range(start_year,start_year+2):
            dynamic_data = extract_yearly_data(year)[batch_start:batch_end]
            combined_data = np.hstack([static_data[batch_start:batch_end], dynamic_data])
            input_batch.append(combined_data)
        input_batch = np.expand_dims(input_batch, axis=-1)
        input_data.append(input_batch)


        for year in range(start_year+1,start_year+3):
            dynamic_data = extract_yearly_target_data(year)[batch_start:batch_end]
            output_batch.append(dynamic_data)
        output_batch = np.expand_dims(output_batch, axis=-1)
        output_data.append(output_batch)

    return np.array(input_data),np.array(output_data)

def create_model(input_shape):
    print(f"Model input shape: {input_shape}")
    model = Sequential()
    model.add(ConvLSTM2D(filters=128, kernel_size=(3, 3), input_shape=input_shape, return_sequences=True, padding='same'))
    model.add(Dropout(0.25))
    model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), return_sequences=True, padding='same'))
    model.add(Dropout(0.25))
    model.add(ConvLSTM2D(filters=1, kernel_size=(3, 3), padding='same'))
    model.add(Flatten())
    model.add(Dense(units=1 * 313 * 1 * 1))
    model.add(Reshape((1, 313, 1, 1)))
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.99, epsilon=None, decay=0.0)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
    return model

# 定义训练模型的函数
def fit_model(model, train_x, train_y, batch_size, valx, valy):
    history = model.fit(train_x, train_y, batch_size=batch_size, epochs=30, validation_data=(valx, valy))
    return history

# 定义预测模型的函数
def predict_model(model, valx):
    return model.predict(valx)

def Calcu_rmse_R2_11_14(cal_flat_true_y, cal_flat_pre_y):
    MSE = mean_squared_error(cal_flat_true_y, cal_flat_pre_y)
    RMSE = np.sqrt(MSE)
    R2 = r2_score(cal_flat_true_y, cal_flat_pre_y)
    return MSE, RMSE, R2


def Flattern(pre_y, true_y):
    cal_true_y = true_y[:, -1:, :, :, :]
    cal_pre_y = pre_y[:, -1:, :, :, :]
    cal_flat_true_y = cal_true_y.reshape(-1)
    cal_flat_pre_y = cal_pre_y.reshape(-1)
    MSE, RMSE, R2 = Calcu_rmse_R2_11_14(cal_flat_pre_y, cal_flat_true_y)
    return MSE, RMSE, R2

static_vars = [
    'Lon', 'Lat', 'rainfall_annual_avg', 'rainfall_summer_avg', 'rainfall_winter_avg',
    'aridity_index', 'PET', 'clay_depth_avg', 'silt_depth_avg', 'sand_depth_avg',
    'regolith_depth', 'LAI', 'NDVI', 'weathering_intensity_index']
dynamic_vars = [
    'Yield_Grid', 'Rainfall_Annual', 'TMax_Annual', 'TMin_Annual', 'AET_Annual', 'Radiation_Annual',
    'Rainfall_Apr_Oct', 'TMax_Apr_Oct', 'TMin_Apr_Oct', 'AET_Apr_Oct', 'Radiation_Apr_Oct',
    'Rainfall_Jun_Aug', 'TMax_Jun_Aug', 'TMin_Jun_Aug', 'AET_Jun_Aug', 'Radiation_Jun_Aug']

# 加载数据
data_path = '...\CAS_CSIRO_Dataset_Wheat_Modelling_20230612_standardized.csv'#替换为您的路径
dataset = pd.read_csv(data_path)

# 生成训练集和验证集的输入变量和目标变量
trainx_dict = {}
trainy_dict = {}
for year in range(2000,2013): #trainx:2000-2013, trainy:2002-2014
    trainx_dict[year], trainy_dict[year] = create_single_input_output_tensor(year)

valx_dict = {}
valy_dict = {}
for year in range(2014,2018): #valx:2014-2017, valy:2016-2019
    valx_dict[year], valy_dict[year] = create_single_input_output_tensor(year)

# 创建模型
input_shape = trainx_dict[2000].shape[1:]
model = create_model(input_shape)

for year in range(2000,2013):
    print(f"Training year:{year}")
    train_x = trainx_dict[year]
    train_y = trainy_dict[year][:,1:,:,:,:]

    fit_model(model, train_x, train_y, batch_size=1)

val_x_2014_2015 = valx_dict[2014]
val_x_2015_2016 = valx_dict[2015]
val_x_2016_2017 = valx_dict[2016]
val_x_2017_2018 = valx_dict[2017]

pre_Yield_Grid_2016 = predict_model(model, val_x_2014_2015)
pre_Yield_Grid_2017 = predict_model(model, val_x_2015_2016)
pre_Yield_Grid_2018 = predict_model(model, val_x_2016_2017)
pre_Yield_Grid_2019 = predict_model(model, val_x_2017_2018)

true__Yield_Grid_2016 = valy_dict[2014][:,1:,:,:,:]
true__Yield_Grid_2017 = valy_dict[2015][:,1:,:,:,:]
true__Yield_Grid_2018 = valy_dict[2016][:,1:,:,:,:]
true__Yield_Grid_2019 = valy_dict[2017][:,1:,:,:,:]

# 展平数组
def flatten_array(arr):
    return arr.reshape(-1)

# 将展平后的数据存入字典
flattened_data = {
    'Yield_Grid_2016': flatten_array(pre_Yield_Grid_2016),
    'Yield_Grid_2017': flatten_array(pre_Yield_Grid_2017),
    'Yield_Grid_2018': flatten_array(pre_Yield_Grid_2018),
    'Yield_Grid_2019': flatten_array(pre_Yield_Grid_2019),
}

# 创建 DataFrame
df = pd.DataFrame(flattened_data)

# 将 DataFrame 存入 Excel 文件
output_path = 'true_Yield_Grid_2016_2019.xlsx'
df.to_excel(output_path, index=False)

