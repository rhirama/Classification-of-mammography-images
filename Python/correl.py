import pandas as pd
import seaborn as sb
import matplotlib.pylab as plt

file_bnd = 'Comparacao_contours54BND.xlsx'
file_edg = 'Comparacao_Contours57EDG.xlsx'

feats_bnd = pd.ExcelFile(file_bnd)
feats_edg = pd.ExcelFile(file_edg)

feats_bnd_05 = feats_bnd.parse('0.05')
feats_bnd_01 = feats_bnd.parse('0.01')
feats_bnd_001 = feats_bnd.parse('0.001')
feats_edg_05 = feats_edg.parse('0.05')
feats_edg_01 = feats_edg.parse('0.01')
feats_edg_001 = feats_edg.parse('0.001')

correl = feats_bnd_05.corr()
sb.heatmap(correl, annot=True)
plt.show()

correl = feats_bnd_01.corr()
sb.heatmap(correl, annot=True)
plt.show()

correl = feats_bnd_001.corr()
sb.heatmap(correl, annot=True)
plt.show()

correl = feats_edg_05.corr()
sb.heatmap(correl, annot=True)
plt.show()

correl = feats_edg_01.corr()
sb.heatmap(correl, annot=True)
plt.show()

correl = feats_edg_001.corr()
sb.heatmap(correl, annot=True)
plt.show()
