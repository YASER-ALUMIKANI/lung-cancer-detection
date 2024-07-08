import os
import pandas as pd
import seaborn as sns
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from numpy import set_printoptions
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


def load_data(filename='cancer_patient_data_sets.csv'):
    """Load the data by detecting the location of the CSV file relative to the script's directory.

    :param filename: Name of the CSV file.
    :return: Pandas dataframe.
    """
    current_directory = os.path.dirname(os.path.abspath(__file__))
    project_root_directory = os.path.abspath(os.path.join(current_directory, '..', '..', '..'))
    data_directory = os.path.join(project_root_directory, 'data')
    file_path = os.path.join(data_directory, filename)

    if os.path.isfile(file_path):
        df = pd.read_csv(file_path)
        return df
    else:
        raise FileNotFoundError(f"{filename} not found in the data directory: {data_directory}")






def new_data(dframe):
    cols = ['Age', 'Air Pollution', 'Alcohol use', 'Dust Allergy', 'OccuPational Hazards', 'Genetic Risk',
        'chronic Lung Disease', 'Balanced Diet', 'Obesity', 'Smoking', 'Passive Smoker', 'Chest Pain',
        'Coughing of Blood', 'Fatigue', 'Wheezing', 'Clubbing of Finger Nails','Frequent Cold', 'Snoring', 'Level']

    new = dframe[cols]
    return new
def convert_to_number(dframe):
    dframe.replace({'Level': {'Low': 0, 'Medium': 1, 'High': 2}}, inplace=True)

def deftop_10_rows(dframe):
    """this function to print top 10 rows.

      :return: None.
      """
    print(dframe.head(10))

def shape_data(dframe):
    """this function to print rows and columns of the dataFrame in tuple.

    :return: None.
    """
    print(dframe.shape)

def type_data(dframe):
    """this function to print the type  of the  data types .

    :return: None.
    """
    print(dframe.dtypes)

def summarize_data(dframe):
    """this function to print Summarize the data using descriptive statistics .
            o Count.
            o Mean.
            o Standard Deviation.
            o Minimum Value.
            o 25th Percentile.
            o 50th Percentile (Median).
            o 75th Percentile.
            o Maximum Value.
    :return: None.
    """
    desc_stats = dframe.describe()
    print(desc_stats)


def levle_class(dframe):
    """this function to  understand the distribution of the level of the cancer .

    :return: None.
    """
    count_level = dframe.groupby(dframe['Level']).size()
    print(count_level)

def correl(dframe):
    """this function to print Correlations Between Attributes
             Correlation refers to the relationship between two variables and how
            they may or may not change together.
             The most common method for calculating correlation is Pearson's
            Correlation.
             A correlation of -1 or 1 shows a full negative or positive correlation
            respectively.
             Whereas a value of 0 shows no correlation at all.
             We can check this by calling corr() method over DataFrame

      :return: None.
      """
    convert_to_number(dframe)
    correlations = dframe.corr(method = 'pearson')[['Level']]     # return dataframe
    eff_corr = correlations[(correlations['Level'] > .5) | (correlations['Level'] < -.5 )]
    print(eff_corr)

def data_histogram(dframe):
    """this function to  polt histogram for the data  .

    :return: None.
    """
    dframe.drop(['index'], axis = 1 ).hist()
    pyplot.show()

def df_density_plt(dframe):
    """this function to  polt density for the data  .
        Density plots are another way of getting a quick idea of the distribution
        of each attribute
       :return: None.
       """
    dframe.drop(['index'], axis=1).plot(
                                kind='density',
                                subplots=True,
                                layout=(5, 5),
                                sharex=False
    )
    pyplot.show()

def pairplot_feature(dframe):
    data = new_data(dframe)
    sns.set_style("white")
    sns.pairplot(data, hue='Level')

def box_whisker_plot(dframe):
    """this function to  polt Box and Whisker  the data  .
        Density plots are another way of getting a quick idea of the distribution
        of each attribute
    :return: None.
    """
    dframe.drop(['index'], axis=1).plot(
        kind='box',
        subplots=True,
        layout=(5, 5),
        sharex=False,
        sharey=False
    )
    pyplot.show()

def rescale_data(arr):
    """this function to rescale the attributes to all have the same scale
       rescaled into the range between 0 and 1.

    :return: None.
    """
    X = arr[:, 2:-1]
    y = arr[:, -1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaledX = scaler.fit_transform(X)
    set_printoptions(precision=3)
    print(rescaledX[0:5, :])

def standardize_data(arr):
    """this function to standardize the data Standardization is a useful technique to transform attributes with a
        Gaussian distribution and differing means and standard deviations to a standard Gaussian distribution with
        a mean of 0 and a standard deviation of 1

    :return: None.
    """
    X = arr[:, 2:-1]
    y = arr[:, -1]
    scaler = StandardScaler().fit(X)
    rescaledX = scaler.fit_transform(X)
    set_printoptions(precision=3)
    print(rescaledX[0:5, :])


def importince_f(dframe):
    """this function to get important features from the existing features

    :return: None.
    """
    convert_to_number(dframe)
    X = dframe.iloc[:, 2:-1]
    y = dframe.iloc[:, -1]
    model = XGBClassifier().fit(X, y)
    df_import_feature = pd.DataFrame(model.feature_importances_)
    df_import_feature.columns = ['values']
    col = list(dframe.columns)[2:-1]
    df_import_feature['name of the columns'] = col
    cond = df_import_feature['values'] > .004
    df_import_feature = df_import_feature[cond]
    print(df_import_feature)

"""def all_models():
    models = {
        'LR': LogisticRegression(max_iter=3000),
        'LDA': LinearDiscriminantAnalysis(),
        'KNN': KNeighborsClassifier(),
        'CART': DecisionTreeClassifier(),
        'NB': GaussianNB(), 'SVM': SVC()
    }
    return models"""

def all_models():
    models = []
    models.append(('LR', LogisticRegression(solver = 'lbfgs', max_iter = 3000)))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))
    return models

def train_models_scores(algorithms, X, y):
    results = []
    names = []

    for name, algorithm in algorithms:
        kfold = KFold(n_splits=10)
        cv_results = cross_val_score(algorithm, X, y, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        msg = (name, cv_results.mean(), cv_results.std())
        print(msg)
    fig = pyplot.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    pyplot.boxplot(results)
    ax.set_xticklabels(names)
    pyplot.show()




























