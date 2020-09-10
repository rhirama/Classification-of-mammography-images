import pandas as pd
import seaborn as sb
import numpy as np
import matplotlib.pylab as plt

file = 'alberta_features.csv'

alberta = pd.read_csv(file, header=0)
alberta = alberta.drop(columns=["img", "diag"])
eps_multipliers = np.linspace(0.001, 0.02, 20)

models = ["_pb", "_opencv"]
for mult in eps_multipliers:
    mult = round(mult, 3)
    for m in models:
        feats_file = "{mult}{model}.csv".format(mult=mult, model=m)
        feats = pd.read_csv(feats_file, header=0, index_col=0)

        feats = pd.concat([feats, alberta], axis=1)

        correl = feats.corr()
        plt.figure(figsize=(16, 10))
        sb.heatmap(correl, annot=True)
        plt.savefig("{mult}{model}.png".format(mult=mult, model=m))
        plt.close()

# # feats_bnd_05 = feats_bnd.parse('0.05')
# feats_bnd_01 = feats_bnd.parse('0.01')
# feats_bnd_001 = feats_bnd.parse('0.001')
# # feats_edg_05 = feats_edg.parse('0.05')
# feats_edg_01 = feats_edg.parse('0.01')
# feats_edg_001 = feats_edg.parse('0.001')

# # correl = feats_bnd_05.corr()
# # sb.heatmap(correl, annot=True)
# # plt.show()

# correl = feats_bnd_01.corr()
# sb.heatmap(correl, annot=True)
# plt.show()

# correl = feats_bnd_001.corr()
# sb.heatmap(correl, annot=True)
# plt.show()

# # correl = feats_edg_05.corr()
# # sb.heatmap(correl, annot=True)
# # plt.show()

# correl = feats_edg_01.corr()
# sb.heatmap(correl, annot=True)
# plt.show()

# correl = feats_edg_001.corr()
# sb.heatmap(correl, annot=True)
# plt.show()
