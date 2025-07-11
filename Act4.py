import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression #Ridge(L1) y Lasso(L2) agregan "regularizacion" : para cuando tengamos Underfitting o Overfitting
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn import set_config
import os
import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import FitFailedWarning, ConvergenceWarning
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from pathlib import Path
from joblib import dump

warnings.filterwarnings("ignore", category=FitFailedWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class ML:
  def __init__(self, X, y, verbose = True, plot = True, regression = None, classification = None, undersample= None, oversample = None, dump= None):
    self.X = X
    self.y = y  
    self.verbose = verbose
    self.plot = plot
    self.regression = regression
    self.classification = classification
    self.undersample = undersample
    self.oversample = oversample
    self.dump = dump

  def dumpfolder(self, file, type = 'model', filename = None):
    output_dir = Path('artefacts')/type
    output_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
      filename = f'{type}.pkl'

    try:
      dump(file, output_dir/filename)
      print(f'{type} se guardo correctamente en {output_dir/filename}')
    except Exception as e:
      print(f'error al guardar {type}: {e}')

  def Preprocess (self):
    X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
    num_cols = self.X.select_dtypes(exclude = 'object').columns
    cat_cols = self.X.select_dtypes(include='object').columns

    preprocessor = ColumnTransformer(
        [
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(), cat_cols)
        ]
    )

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    if self.dump:
      self.dumpfolder(preprocessor, type = 'preprocessor', filename = 'preprocessor.pkl')
    
    if self.classification:

      le = LabelEncoder()   #nombra a los tipos (de carros por ejemplo) con numeros para que le sea mas facil

      y_train = le.fit_transform(y_train)
      y_test = le.transform(y_test)

      if self.dump:
        self.dumpfolder(le, type = 'preprocessor', filename = 'labelencoder.pkl')

    elif self.regression:
      pass

    if self.oversample:
      smote = SMOTE(random_state=42, sampling_strategy = 'minority')
      X_train, y_train = smote.fit_resample(X_train, y_train)

    elif self.undersample:
      rus = RandomUnderSampler(random_state=42)
      X_train, y_train = rus.fit_resample(X_train, y_train)


    # Disminuir datos de las clases (del support) para que esten parejas todas, solo  en los datos del TRAIN, no en los del test (a ese no se le hace nada)
    rus = RandomUnderSampler(random_state=42)
    X_train, y_train = rus.fit_resample(X_train, y_train)

    '''
    # Generar datos para que esten parejas las clases, datos del TRAIN
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    '''

    return X_train, X_test, y_train, y_test

  def LogReg(self, X_train, X_test, y_train, y_test):
    lr = LogisticRegression()
    lr_grid = [
        {
            'penalty': ['l1', 'l2', 'none'],
            'C': [0.01, 0.1, 1, 10, 100],
            'max_iter': [100, 1000, 2000],
        }
    ]

    grid_search = GridSearchCV(lr, lr_grid, cv=5, verbose = 1)
    grid_search.fit(X_train, y_train)

    grid_best_params = grid_search.best_params_
    lrfinal = LogisticRegression(**grid_best_params)
    lrfinal.fit(X_train, y_train)

    if self.dump:
      self.dumpfolder(lrfinal, type = 'model', filename = 'logreg.pkl')

    y_pred = lrfinal.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    if self.verbose:
      print('\n')
      print('--------------------------------\n')
      print('Regresion Logistica\n')
      print(f'Accuracy: {accuracy}')
      #Desgloce de como  clasifico cada clase con su accuracy
      class_report = classification_report(y_test, y_pred)
      print(class_report)

    if self.plot:
    
      #Matriz de confision, REALIZAR PARA SABER SI EL MODELO FUNCIONA BIEN
      cm = confusion_matrix(y_test, y_pred)
      disp = ConfusionMatrixDisplay(confusion_matrix=cm)
      disp.plot(cmap='Blues')
      plt.show()               # Mostrar siempre para ver como el modelo esta clasificando los datos y saber que tan bueno es el modelo, no solo con el accuracy

  def SVMLin(self, X_train, X_test, y_train, y_test):

    svc = LinearSVC()

    svc_grid = [
        {
            'C': [0.01, 0.1, 1, 10, 100],
            'class_weight': [None, 'balanced'],
            'fit_intercept': [True, False],
            'penalty': ['l1', 'l2', 'none'],
        }
    ]

    grid_search = GridSearchCV(svc, svc_grid, cv=5, verbose = 1)
    grid_search.fit(X_train, y_train)

    grid_best_params = grid_search.best_params_
    svcfinal = LinearSVC(**grid_best_params)
    svcfinal.fit(X_train, y_train)

    if self.dump:
      self.dumpfolder(svcfinal, type = 'model', filename = 'svc.pkl')

    y_pred = svcfinal.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    if self.verbose:
      print('\n')
      print('--------------------------------\n')
      print('SVM Lineal\n')
      print(f'Accuracy: {accuracy}')
      class_report = classification_report(y_test, y_pred)
      print(class_report)

    if self.plot:
      cm = confusion_matrix(y_test, y_pred)
      disp = ConfusionMatrixDisplay(confusion_matrix=cm)
      disp.plot(cmap='Blues')
      plt.show()               

  def SVM(self, X_train, X_test, y_train, y_test):

    svc = SVC()

    svc_grid = [
        {
            'C': [0.01, 0.1, 1, 10, 100],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'gamma': ['scale', 'auto'],
        }
    ]

    grid_search = GridSearchCV(svc, svc_grid, cv=5, verbose = 1)
    grid_search.fit(X_train, y_train)

    grid_best_params = grid_search.best_params_
    svcfinal = SVC(**grid_best_params)
    svcfinal.fit(X_train, y_train)

    
    if self.dump:
      self.dumpfolder(svcfinal, type = 'model', filename = 'svc.pkl')

    y_pred = svcfinal.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    if self.verbose:
      print('\n')
      print('--------------------------------\n')
      print('SVM\n')
      print(f'Accuracy: {accuracy}')
      class_report = classification_report(y_test, y_pred)
      print(class_report)

  
    if self.plot:
      cm = confusion_matrix(y_test, y_pred)
      disp = ConfusionMatrixDisplay(confusion_matrix=cm)
      disp.plot(cmap='Blues')
      plt.show()               
  
  def DecisionTree(self, X_train, X_test, y_train, y_test):

    DTC = DecisionTreeClassifier()

    DTC_grid = [
        {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'class_weight': [None, 'balanced'],
        }
    ]

    grid_search = GridSearchCV(DTC, DTC_grid, cv=5, verbose = 1)
    grid_search.fit(X_train, y_train)

    grid_best_params = grid_search.best_params_
    DTCfinal = DecisionTreeClassifier(**grid_best_params)
    DTCfinal.fit(X_train, y_train)

    if self.dump:
      self.dumpfolder(DTCfinal, type = 'model', filename = 'dtc.pkl')

    y_pred = DTCfinal.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    if self.verbose:
      print('\n')
      print('--------------------------------\n')
      print('DTC\n')
      print(f'Accuracy: {accuracy}')
      class_report = classification_report(y_test, y_pred)
      print(class_report)

    if self.plot:
      cm = confusion_matrix(y_test, y_pred)
      disp = ConfusionMatrixDisplay(confusion_matrix=cm)
      disp.plot(cmap='Blues')
      plt.show()

  '''
    #Arbol de decision

    from six import StringIO
    from IPython.display import Image, display
    from sklearn.tree import export_graphviz
    import pydotplus

    dot_data = StringIO()
    export_graphviz(DTCfinal, out_file=dot_data, filled=True, rounded=True, special_characters=True)

    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    display(Image(graph.create_png()))  # Aquí sí se muestra en la celda
    graph.write_png('arbol.png')
  '''

  def RandomForest(self, X_train,X_test, y_train, y_test):

    RFC = RandomForestClassifier()

    RF_grid = [
        {
            'criterion': ['gini', 'entropy'],
            'class_weight': ['balanced', 'balanced_subsample', None],
            'warm_start': [True, False]
        }
    ]

    grid_search = GridSearchCV(RFC, RF_grid, cv=5, verbose = 1)
    grid_search.fit(X_train, y_train)

    grid_best_params = grid_search.best_params_
    RFCfinal = RandomForestClassifier(**grid_best_params)
    RFCfinal.fit(X_train, y_train)

    if self.dump:
      self.dumpfolder(RFCfinal, type = 'model', filename = 'rfc.pkl')

    y_pred = RFCfinal.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    if self.verbose:
      print('\n')
      print('--------------------------------\n')
      print('Random Forest\n')
      print(f'Accuracy: {accuracy}')
      class_report = classification_report(y_test, y_pred)
      print(class_report)

    if self.plot:
      cm = confusion_matrix(y_test, y_pred)
      disp = ConfusionMatrixDisplay(confusion_matrix=cm)
      disp.plot(cmap='Blues')
      plt.show()

  def AdaBoost(self, X_train,X_test, y_train, y_test):

    AB = AdaBoostClassifier()

    AB_grid = [
        {
            'n_estimators': [10, 50, 100],
            'learning_rate': [0.1, 0.5, 1],
            'algorithm': ['SAMME', 'SAMME.R']
        }
    ]

    grid_search = GridSearchCV(AB, AB_grid, cv=5, verbose = 1)
    grid_search.fit(X_train, y_train)

    grid_best_params = grid_search.best_params_
    ABfinal = AdaBoostClassifier(**grid_best_params)
    ABfinal.fit(X_train, y_train)

    if self.dump:
      self.dumpfolder(ABfinal, type = 'model', filename = 'ab.pkl')

    y_pred = ABfinal.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    if self.verbose:
      print('\n')
      print('--------------------------------\n')
      print('AdaBoost\n')
      print(f'Accuracy: {accuracy}')
      class_report = classification_report(y_test, y_pred)
      print(class_report)

    if self.plot:
      cm = confusion_matrix(y_test, y_pred)
      disp = ConfusionMatrixDisplay(confusion_matrix=cm)
      disp.plot(cmap='Blues')
      plt.show()

  def GradientBoost(self, X_train,X_test, y_train, y_test):

    GBC = GradientBoostingClassifier()

    GBC_grid = [
        {
            'n_estimators': [10, 50, 100],
            'learning_rate': [0.1, 0.5, 1],
            'subsample': [0.8, 1.0],
            #'max_depth': [3, 5, 7]
        }
    ]

    grid_search = GridSearchCV(GBC, GBC_grid, cv=5, verbose = 1)
    grid_search.fit(X_train, y_train)

    grid_best_params = grid_search.best_params_
    GBCfinal = GradientBoostingClassifier(**grid_best_params)
    GBCfinal.fit(X_train, y_train)

    if self.dump:
      self.dumpfolder(GBCfinal, type = 'model', filename = 'gbc.pkl')

    y_pred = GBCfinal.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    if self.verbose:
      print('\n')
      print('--------------------------------\n')
      print('GradientBoost\n')
      print(f'Accuracy: {accuracy}')
      class_report = classification_report(y_test, y_pred)
      print(class_report)

    if self.plot:
      cm = confusion_matrix(y_test, y_pred)
      disp = ConfusionMatrixDisplay(confusion_matrix=cm)
      disp.plot(cmap='Blues')
      plt.show()

  def XGBoost(self, X_train,X_test, y_train, y_test):

    XGB = xgb.XGBClassifier()

    XGB_grid = [
        {
            'n_estimators': [10, 50, 100],
            'learning_rate': [0.1, 0.5, 1],
            'subsample': [0.8, 1.0],
            #'max_depth': [3, 5, 7]
        }
    ]

    grid_search = GridSearchCV(XGB, XGB_grid, cv=5, verbose = 1)
    grid_search.fit(X_train, y_train)

    grid_best_params = grid_search.best_params_
    XGBfinal = xgb.XGBClassifier(**grid_best_params)
    XGBfinal.fit(X_train, y_train)

    if self.dump:
      self.dumpfolder(XGBfinal, type = 'model', filename = 'xgb.pkl')

    y_pred = XGBfinal.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    if self.verbose:
      print('\n')
      print('--------------------------------\n')
      print('XG Booost\n')
      print(f'Accuracy: {accuracy}')
      class_report = classification_report(y_test, y_pred)
      print(class_report)

    if self.plot:
      cm = confusion_matrix(y_test, y_pred)
      disp = ConfusionMatrixDisplay(confusion_matrix=cm)
      disp.plot(cmap='Blues')
      plt.show()

  def Run(self):
    X_train, X_test, y_train, y_test = self.Preprocess()
    self.LogReg(X_train, X_test, y_train, y_test)
    self.SVMLin(X_train, X_test, y_train, y_test)
    self.SVM(X_train, X_test, y_train, y_test)
    self.DecisionTree(X_train, X_test, y_train, y_test)
    self.RandomForest(X_train, X_test, y_train, y_test)
    self.AdaBoost(X_train, X_test, y_train, y_test)
    self.GradientBoost(X_train, X_test, y_train, y_test)
    self.XGBoost(X_train, X_test, y_train, y_test)


'''
df = pd.read_csv('/content/drive/MyDrive/used_car_price_dataset_extended.csv')
#print(df.head())

X = df.drop(['transmission'], axis=1)
y = df['transmission']

X_train, X_test, y_train, y_test = Preprocess(X,y)

LogReg(X_train, X_test, y_train, y_test)
'''

test_pd = pd.read_csv(r'C:\Users\Dell\Documents\Curso de Verano\test.csv')
train_pd = pd.read_csv(r'C:\Users\Dell\Documents\Curso de Verano\train.csv')
df2 = pd.concat([test_pd, train_pd])
#print(df2.head())

df2.drop(['id'], axis = 1, inplace = True)
df2.dropna(inplace = True)

X = df2.drop(['price_range'], axis=1)
y = df2['price_range']

MLTrain = ML(X, y, classification = True, dump = True, plot = False, verbose = True)
MLTrain.Run()

