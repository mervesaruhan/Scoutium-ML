
# İş Problemi
# Scout’lar tarafından izlenen futbolcuların özelliklerine verilen puanlara göre, oyuncuların hangi sınıf
# (average, highlighted) oyuncu olduğunu tahminleme.

###################################################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV

#pip install yellowbrick


import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_validate

from sklearn.metrics import RocCurveDisplay

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)


pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)






# Adım1: scoutium_attributes.csv ve scoutium_potential_labels.csv dosyalarını okutunuz.

df_att=pd.read_csv(r'C:\Users\SARU\Desktop\VBO\Machine Learning\ML-III\Case Study-V\scoutium_attributes.csv',sep=';')
df_pot=pd.read_csv(r'C:\Users\SARU\Desktop\VBO\Machine Learning\ML-III\Case Study-V\scoutium_potential_labels.csv',sep=';')

df_pot.columns
df_att.columns

# Adım2: Okutmuş olduğumuz csv dosyalarını merge fonksiyonunu kullanarak birleştiriniz.
# ("task_response_id", 'match_id', 'evaluator_id' "player_id" 4 adet değişken üzerinden birleştirme işlemini gerçekleştiriniz.)


df_merge =pd.merge(df_att, df_pot, how='right', on=['task_response_id', "match_id","evaluator_id","player_id"])
df=df_merge.copy()



#sep eklemediğim zaman merge kısmında hata veriyor!

'''#Birleştirme hatası, iki DataFrame'deki birleştirme sütunlarının belirtilen isimlere sahip olduğunu varsayarak gerçekleştirildiğinden,
verilerin doğru şekilde birleştirilebilmesi için aynı ayırıcı karakterinin kullanılması önemlidir. sep parametresinin doğru şekilde
belirtilmemesi, sütunların yanlış şekilde ayrılmasına ve hataların oluşmasına neden olabilir.'''



# Adım3: position_id içerisindeki Kaleci (1) sınıfını veri setinden kaldırınız.

df.head()
df.info()
df.isnull().sum()

#genel resim
def check_df(dataframe):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(3))
    print("##################### Tail #####################")
    print(dataframe.tail(3))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
check_df(df)

#eksik degerler
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns
missing_values_table(df)

#korelasyon
corr = df.corr()
corr
sns.set(rc={'figure.figsize': (12, 12)})
sns.heatmap(corr, cmap="RdBu")
plt.show()

sns.heatmap(corr, annot=True, cmap='RdBu')

##eksik değişkenlerin veri çerisindeki oranına bakmak
na_rows = [col for col in df.columns if df[col].isnull().any()]
def missing_value_ratio(dataframe,na_rows):
    print({'Ratio':(df[na_rows].isnull().sum()*100/dataframe.shape[0]).sort_values(ascending=False)})
missing_value_ratio(df,na_rows)




df.drop(df[df['position_id'] == 1].index, inplace = True)


'''index parametresi kullanılarak seçilen satırların dizin numaraları (indeksleri) alınır ve drop()
fonksiyonu ile bu satırlar DataFrame'den çıkarılır.'''


# Adım4: potential_label içerisindeki below_average sınıfını veri setinden kaldırınız.( below_average sınıfı tüm verisetinin %1'ini oluşturur)


below_average_count = df['potential_label'].value_counts()['below_average']*100/df.shape[0]    # ---> veri seti içerisindeki oranına bakma

df.drop(df[df['potential_label'] == 'below_average'].index, inplace = True)


# Adım5: Oluşturduğunuz veri setinden “pivot_table” fonksiyonunu kullanarak bir tablo oluşturunuz. Bu pivot table'da her satırda bir oyuncu
# olacak şekilde manipülasyon yapınız.

output = pd.pivot_table(data=df,
                        index=['player_id','position_id','potential_label'],
                        columns=['attribute_id'],
                        values='attribute_value'
                        )

output.head()

# Adım6: Label Encoder fonksiyonunu kullanarak “potential_label” kategorilerini (average, highlighted) sayısal olarak ifade ediniz.

df['potential_label'].unique()

output.info()
output.reset_index(inplace=True)

#output = output.astype(str)
output.head()
output.columns

output.columns = output.columns.map(str)

def label_encoder(df, column):
    labelencoder = LabelEncoder()
    df[column] = labelencoder.fit_transform(df[column])
    return df

output = label_encoder(output, 'potential_label')
output.tail()


# Adım7: Sayısal değişken kolonlarını “num_cols” adıyla bir listeye atayınız.


num_cols=output.columns[3:]




# Adım8: Kaydettiğiniz bütün “num_cols” değişkenlerindeki veriyi ölçeklendirmek için StandardScaler uygulayınız.

ss = StandardScaler()
output[num_cols] = ss.fit_transform(output[num_cols])


# Adım9: Elimizdeki veri seti üzerinden minimum hata ile futbolcuların potansiyel etiketlerini tahmin eden bir makine öğrenmesi modeli
# geliştiriniz. (Roc_auc, f1, precision, recall, accuracy metriklerini yazdırınız.)

y = output["potential_label"]
X = output.drop(["potential_label", "player_id"], axis=1)


X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    random_state = 123,
                                                    stratify = y,
                                                    test_size = 0.2,
                                                    shuffle = True)

print(f"The shape of X_train is --> {colored(X_train.shape,'red')}")
print(f"The shape of X_test is  --> {colored(X_test.shape,'red')}")
print(f"The shape of y_train is --> {colored(y_train.shape,'red')}")
print(f"The shape of y_test is  --> {colored(y_test.shape,'red')}")




def classification_models(model):
    y_pred = model.fit(X_train, y_train).predict(X_test)
    accuracy = accuracy_score(y_pred, y_test)
    roc_score = roc_auc_score(y_pred, model.predict_proba(X_test)[:, 1])
    f1 = f1_score(y_pred, y_test)
    precision = precision_score(y_pred, y_test)
    recall = recall_score(y_pred, y_test)

    results = pd.DataFrame({"Values": [accuracy, roc_score, f1, precision, recall],
                            "Metrics": ["Accuracy", "ROC-AUC", "F1", "Precision", "Recall"]})

    # Visualize Results:
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Bar(x=[round(i, 5) for i in results["Values"]],
                         y=results["Metrics"],
                         text=[round(i, 5) for i in results["Values"]], orientation="h", textposition="inside",
                         name="Values",
                         marker=dict(color=["indianred", "firebrick", "palegreen", "skyblue", "plum"],
                                     line_color="beige", line_width=1.5)), row=1, col=1)
    fig.update_layout(title={'text': model.__class__.__name__,
                             'y': 0.9,
                             'x': 0.5,
                             'xanchor': 'center',
                             'yanchor': 'top'},
                      template='plotly_white')
    fig.update_xaxes(range=[0, 1], row=1, col=1)

    iplot(fig)

import plotly.graph_objs as go
from plotly.subplots import make_subplots

import joblib
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler


my_models = [
    LogisticRegression(),
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    GradientBoostingClassifier(),
]

for model in my_models:
    classification_models(model)


# Adım10: Değişkenlerin önem düzeyini belirten feature_importance fonksiyonunu kullanarak özelliklerin sıralamasını çizdiriniz.


def plot_importance(model, features, num=len(X), save=False):

    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.png")

model = GradientBoostingClassifier()
model.fit(X, y)

plot_importance(model, X)


#############################################################################

###################NOTLAR#########################################


#hiyerarşikte de kmanste de modein cıkardıgı sonuçlara direkt güvenemiyoruz cünkü
# bu yöntemler matemaiksel benzerliklere dayalıdır. Buradaki matematiksel benzerlikler teorik olarak aynı davranısı göstermiyor olabilir.
#diğer gözetimli ögrenmelerde model ve iktisat degerileri de dahildir. varsayımlar araısnda bir bağıntı bir ilişki var ve sebep sonuç noktasında
# yönlendiriciliği var. Bir teoriye dayalıdır.unsupervised yöntemlerde iş biligisi ön planda tutulmalıdır.

#PCA:temelde veri seri içierinisndeki bilginin bir miktarının kaybolsını göze alarak veri setini daha az sayıdaki bileşen ile tmesil eidyor.
#


#HR Sorusu: Sunumlarınızı nasıl geliştirirsiniz?(teknik geçmişi olan ve olmayan)
#en büyük problem muhatapların konuyla ilgili hiçbir şey bimedigi varsayımalı.
#teknik olmayana konu en basit dille anatılmal teknik jargondan uzak bir şekilde.
# Örneklendirme yapılmalı. hikayeleştrime yapılmalı ki nisanlar dinlerken keyif alsın.
#merak uyandırmak önemlidir. Beklenti yönetimi yapılmalıdır. Belnetilerin farkında olundugu karsı tarafa hissettirilmeli.
#kpnuyu toparlamak,

##PCA Nedir,hangi amaçlar için kullanılır?
#ABC Dİr. iki üç bileşenle temsil. 