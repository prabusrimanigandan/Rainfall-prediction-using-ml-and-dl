import chart_studio.plotly as py
import chart_studio.tools as tls
import chart_studio
import plotly.express as px
from pycaret.classification import *
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()
tls.set_credentials_file(os.getenv('USERID'), os.getenv('APIKEY'))


def regTest(src):
    test_data = pd.read_csv(src)

    model1 = tf.keras.models.load_model(
        r'C:\Users\Admin\Documents\5_6174891034263159160\Rainfall_prediction\src\model\neuroregress(TN)model', compile=True)
    month = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
             'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    test_list = []

    for i in test_data.columns:
        if i in month:
            test_list.append(i)
    updatedtest_data = test_data[test_list]
    pred_res = model1.predict(updatedtest_data)
    test_data['Label'] = pred_res
    return test_data


def classyTest(src):
    test_data = pd.read_csv(src)
    model_loc = r'C:\Users\Admin\Documents\5_6174891034263159160\Rainfall_prediction\src\model\Rainfallclassymodel'
    prediction = load_model(model_loc)
    predict_res = predict_model(estimator=prediction, data=test_data)
    return predict_res


def save_csv(dataframe, loc):
    dataframe.to_csv(loc, index=False)
    return ("Saved at: ", loc)


def reg_plot(dataframe):
    plot1 = px.line(dataframe, x=dataframe.index,
                    y=dataframe['Label'], title='predicted result')
    fig1 = py.plot(plot1, filename='predicted result', auto_open=False)
    return tls.get_embed(fig1)


def pie_plot(dataframe):
    label_info = pd.DataFrame(dataframe['Label'].value_counts(
    ).reset_index().values, columns=["Label", "No. of Data"])
    plot1 = px.pie(label_info, names=label_info['Label'],
                   values=label_info['No. of Data'], title="Prediction summary")
    fig = py.plot(plot1, filename="Prediction result", auto_open=False)
    return(tls.get_embed(fig))


def bar_plot(dataframe):
    label_info = pd.DataFrame(dataframe['Label'].value_counts(
    ).reset_index().values, columns=["Label", "No. of Data"])
    plot1 = px.bar(label_info, x=label_info['Label'],
                   y=label_info['No. of Data'], title="Prediction summary")
    fig1 = py.plot(plot1, filename="Prediction summary", auto_open=False)
    return tls.get_embed(fig1)
