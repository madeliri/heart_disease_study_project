# импотр библиотек
import joblib
import pandas as pd

# загрузка подготовленных ранее модели и скейлера
model = joblib.load('models/model.joblib')
scaler = joblib.load('models/scaler.joblib')

# загрузка тестовых данных
df = pd.read_csv('data/test.csv')

# удаляем столбец c id пациента (он совершенно неинформативен)
df = df.drop('ID',axis=1)


# задаем вручную список категориальных переменных
cat_list = ([
  'sex',
  # 'chest', # все таки трудно считать его категориальной переменной
  'fasting_blood_sugar',
  'resting_electrocardiographic_results',
  'exercise_induced_angina',
  'slope',
  'number_of_major_vessels',
  'thal'
  ])

# выделяем отдельно столбцы с количественными переменными
num_cols = df.select_dtypes(include=['number']).columns

# циклом переводим в тип object
for elem in cat_list:
  df[elem] = df[elem].astype(object)

# перевод категориальных переменных в дамми-переменные
df_mod = pd.get_dummies(
    df,
    columns=cat_list,
    dtype='int',
    drop_first=True
)

# преобразуем датасет
df_scaled = scaler.transform(df_mod)

# и предсказываем параметры
predict = model.predict(df_scaled)

print(predict)