from sklearn.preprocessing import MinMaxScaler
import pandas
import openpyxl
import xlsxwriter
import os
import matplotlib.pyplot as plt


files_paths = ['D:\\Users\\Rodrigo S. Hirama\\Documentos\\EACH\\IC\\Comparacao_contours54BND.xlsx', 'D:\\Users'
                                                                                                    '\\Rodrigo S. '
                                                                                                    'Hirama\\Documentos\\EACH\\IC\\Comparacao_contours57EDG.xlsx']
sheet_names = ['0.05', '0.01', '0.001']
writer = None
features = None

for file in files_paths:
    for name in sheet_names:
        features = pandas.read_excel(file, sheet_name=name, header=0, skipfooter=1)
        mms = MinMaxScaler(feature_range=(0, 1))
        norm_si = mms.fit_transform(features[['spiculation_index']])
        norm_si_2 = mms.fit_transform(features[['spiculation_index_z']])
        features['normalized_si'] = norm_si
        features['normalized_si_z'] = norm_si_2
        print(features)
        file_name = os.path.basename(file)
        excel = openpyxl.load_workbook(file_name, read_only=False)
        writer = pandas.ExcelWriter(file_name, engine='openpyxl')
        writer.book = excel
        writer.sheets = dict((ws.title, ws) for ws in excel.worksheets)
        features.to_excel(writer, sheet_name=name, index=False)
        writer.save()
writer.close()

