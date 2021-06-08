import os
import shutil
from flask import *
import pandas as pd
from werkzeug.utils import secure_filename
import module as md
from dotenv import load_dotenv, main
load_dotenv()

app = Flask(__name__, template_folder='templates')
upload_folder = os.getenv('UPLOAD_FOLDER')
data_folder = os.getenv('DATA_FOLDER')


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/prediction")
def prediction():
    return render_template('prediction.html')


@app.route("/upload", methods=['POST'])
def upload():
    f = request.files['upload_monthfile']
    filename = secure_filename(f.filename)
    f.save(os.path.join(
        r'C:\Users\Admin\Documents\5_6174891034263159160\Rainfall_prediction\upload_folder', filename))
    src_file = os.path.join(
        r'C:\Users\Admin\Documents\5_6174891034263159160\Rainfall_prediction\upload_folder', filename)
    dest_file = os.path.join(data_folder, 'data.csv')
    # print(os.getenv('NEUROTN_MODEL'))
    shutil.move(src_file, dest_file)
    prediction_result = md.regTest(dest_file)
    print(prediction_result.head())
    md.save_csv(prediction_result, (os.path.join(
        data_folder, 'predicted_data.csv')))
    data_length = len(prediction_result['Label'])
    plot0 = md.reg_plot(prediction_result)
    render_content = {
        'filename': filename,
        'data_length': data_length,
        'plot0': plot0
    }
    return render_template('report.html', **render_content)


@app.route("/dayupload", methods=['POST'])
def dayupload():
    f = request.files['upload_dayfile']
    filename = secure_filename(f.filename)
    f.save(os.path.join(
        r'C:\Users\Admin\Documents\5_6174891034263159160\Rainfall_prediction\upload_folder', filename))
    src_file = os.path.join(
        r'C:\Users\Admin\Documents\5_6174891034263159160\Rainfall_prediction\upload_folder', filename)
    dest_file = os.path.join(data_folder, 'data.csv')
    shutil.move(src_file, dest_file)
    prediction_result = md.classyTest(dest_file)

    md.save_csv(prediction_result, (os.path.join(
        data_folder, 'predicted_data.csv')))
    data_length = len(prediction_result['Label'])
    plot0 = md.pie_plot(prediction_result)
    #plot1 = md.bar_plot(prediction_result)
    print(plot0)
    render_content = {
        'filename': filename,
        'data_length': data_length,
        'plot0': plot0
    }
    return render_template('report.html', **render_content)


@app.route("/download/csv")
def download():
    return send_from_directory(directory=data_folder, path='predicted_data.csv', as_attachment=True)


if __name__ == '__main__':
    app.run(host='localhost', port=os.getenv('PORT'), debug=True)
