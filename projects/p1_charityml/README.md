# Supervised Learning (binary classification for an imbalanced data) 
## using [scikit-learn](http://scikit-learn.org/stable/supervised_learning.html)
This project is for the CharityML project for [the Udacity course: Intro to Machine Learning with TensorFlow](https://www.udacity.com/course/intro-to-machine-learning-with-tensorflow-nanodegree--nd230), and [its github repository](https://github.com/udacity/intro-to-ml-tensorflow).  


### Goal
Classify people based on the features explained below, to predict their income class, either above 50K or below 50K. This can be used to identigy possible donors to a charity. 
### Data
The modified census dataset consists of approximately 32,000 data points, with each datapoint having 13 features. This dataset is a modified version of the dataset published in the paper *"Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid",* by Ron Kohavi. You may find this paper [online](https://www.aaai.org/Papers/KDD/1996/KDD96-033.pdf), with the original dataset hosted on [UCI](https://archive.ics.uci.edu/ml/datasets/Census+Income).

**Features**
- `age`: Age
- `workclass`: Working Class (Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked)
- `education_level`: Level of Education (Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool)
- `education-num`: Number of educational years completed
- `marital-status`: Marital status (Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse)
- `occupation`: Work Occupation (Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces)
- `relationship`: Relationship Status (Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried)
- `race`: Race (White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black)
- `sex`: Sex (Female, Male)
- `capital-gain`: Monetary Capital Gains
- `capital-loss`: Monetary Capital Losses
- `hours-per-week`: Average Hours Per Week Worked
- `native-country`: Native Country (United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands)

**Target Variable**
- `income`: Income Class (<=50K, >50K)

### Install

This project requires **Python 3.x** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)

You will also need to have software installed to run and execute an [iPython Notebook](http://ipython.org/notebook.html)

We recommend students install [Anaconda](https://www.continuum.io/downloads), a pre-packaged Python distribution that contains all of the necessary libraries and software for this project.

### Code

The code is provided in the `finding_donors.ipynb` notebook file. the `visuals.py` Python file and the `census.csv` dataset file is used. 

### Run

In a terminal or command window, navigate to the top-level project directory `finding_donors/` (that contains this README) and run one of the following commands:

```bash
ipython notebook finding_donors.ipynb
```  
or
```bash
jupyter notebook finding_donors.ipynb
```

This will open the iPython Notebook software and project file in your browser.

### Pipeline 
My pipeline consist of the follwoing steps, and the code is called `finding_donors.ipynb`. I used the following libraries

1. Importing data
```
data = pd.read_csv("census.csv")
```
2. Exploring the data, e.g. computing the number records for each output class 
```
display(data.head(n=5))
n_records = len(data.index)
n_at_most_50k, n_greater_50k = data.income.value_counts()

import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(data['income'] )
```

2. Preprocessing the data
   1. Removing redundant data
   ```
   features_raw = data.drop('education_level', axis = 1)
   ```
   2. Transforming Skewed Continuous Features
   ```
   features_log_transformed = pd.DataFrame(data = features_raw)
   features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))
   ```
   3. Normalizing Numerical Features
   ```from sklearn.preprocessing import MinMaxScaler

   # Initialize a scaler, then apply it to the features
   scaler = MinMaxScaler() # default=(0, 1)
   # select numerical data 
   numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
   
   features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)
   features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])
   ```
   4. one-hot encoding
   ```
   income = income_raw.apply(lambda x: 0 if x == '<=50K' else 1)
   features_final = pd.get_dummies(features_log_minmax_transform)
   ```
   5. examine if there is any null dat
   ```
   features_final.isnull().sum()
   income.isnull().sum()
   ```
   5. Shuffle and Split Data into training and testing
   ```
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(features_final, 
                                                    income, 
                                                    test_size = 0.2, 
                                                    random_state = 42)
    ```
3. Training and Predicting
  1. Useing fbeta_score and accuracy_score
   from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
   ```
   def train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 
    results = {}
    # Fit the learner to the training data using slicing with 'sample_size' 
    learner.fit(X_train[:sample_size], y_train[:sample_size])
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    results['acc_train'] = accuracy_score(y_train[:300],predictions_train)
    results['acc_test'] = accuracy_score(y_test,predictions_test)
    results['f_train'] = fbeta_score(y_train[:300],predictions_train, beta=0.5)
    results['f_test'] = fbeta_score(y_test,predictions_test, beta=0.5)
    return results
   ```
  2. Initial Model Evaluation
  ```
   from sklearn.tree import DecisionTreeClassifier
   from sklearn.ensemble import  RandomForestClassifier
   from sklearn.svm import SVC

   random_state=100

   clf_B = DecisionTreeClassifier(random_state=random_state, max_depth=10 ,min_samples_leaf=15 )
   clf_d = RandomForestClassifier(random_state=random_state, max_depth=10 ,min_samples_leaf=15 )
   clf_i = SVC(random_state=random_state)

   # Collect results on the learners
   results = {}
   for clf in [clf_B, clf_d, clf_i]:
       clf_name = clf.__class__.__name__
      results[clf_name] = {}
      for i, samples in enumerate([samples_1, samples_10, samples_100]):
         results[clf_name][i] = \
         train_predict(clf, samples, X_train, y_train, X_test, y_test)
   ```
  3. Improving models (Model Tuning)
      1. using grid search (GridSearchCV)
      ```
      from sklearn.model_selection import GridSearchCV
      from sklearn.metrics import make_scorer
      from sklearn.model_selection import GridSearchCV
      from sklearn.metrics import make_scorer
      clf_3 = DecisionTreeClassifier(random_state=42)

      parameters_3 = {'max_depth':[2,4,6,8,10,15],'min_samples_leaf':[2,4,6,8,10,15], 'min_samples_split':[2,4,6,8,10,15]}
      scorer = make_scorer(fbeta_score,beta=0.5)

      # TODO: Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()
      grid_obj = GridSearchCV(clf, param_grid=parameters, scoring = scorer)

      # TODO: Fit the grid search object to the training data and find the optimal parameters using fit()
      grid_fit = grid_obj.fit(X_train, y_train)

      # Get the estimator
      best_clf = grid_fit.best_estimator_

      # Make predictions using the unoptimized and model
      predictions = (clf.fit(X_train, y_train)).predict(X_test)
      best_predictions = best_clf.predict(X_test)
      ```
      2. confusion matrix
      ```
      from sklearn.metrics import confusion_matrix
      confusion_matrix(y_test,best_predictions)
      pd.crosstab(y_test, best_predictions, rownames = ['Actual'], colnames =['Predicted'], margins = True)
      ```   
      3. Feature Relevance Observation
      ```
      sns.set(style="ticks")
      features_final.describe()

      selected_columns = features_final[['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']]
      new_df1 = selected_columns.copy()
      new_df2 = pd.concat([new_df1, income], axis=1, sort=False)

      sns.heatmap(new_df2.corr(), annot=True, cmap="YlGnBu");

      sns.pairplot(new_df2, hue = 'income');

      new_df2.hist();

      ```
      and
      ```
      learner = AdaBoostClassifier()

      learner.fit(X_train, y_train)
      predictions_test = learner.predict(X_test)

      features = features_final.columns[:features_final.shape[1]]

      importances = learner.feature_importances_
      indices = np.argsort(importances)
      ```
      4. Feature Importance: i.e. using only the most important features such that the accuracy does not decrease
      ```
      from sklearn.base import clone

      X_train_reduced = X_train[X_train.columns.values[(np.argsort(importances)[::-1])[:5]]]
      X_test_reduced = X_test[X_test.columns.values[(np.argsort(importances)[::-1])[:5]]]

      clf = (clone(best_clf)).fit(X_train_reduced, y_train)

      reduced_predictions = clf.predict(X_test_reduced)
      ```
      or 
      ```
      X_train_reduced = SelectPercentile(chi2, percentile=10).fit_transform(X_train, y_train)
      X_test_reduced = SelectPercentile(chi2, percentile=10).fit_transform(X_test, y_test)
      clf = (clone(best_clf)).fit(X_train_reduced, y_train)
      reduced_predictions = clf.predict(X_test_reduced)
      ```
      or
      ```
      from sklearn.feature_selection import SelectKBest, chi2
      X_train_reduced = SelectKBest(chi2,  k=20).fit_transform(X_train, y_train)
      X_test_reduced = SelectKBest(chi2,  k=20).fit_transform(X_test, y_test)
      clf = (clone(best_clf)).fit(X_train_reduced, y_train)
      reduced_predictions = clf.predict(X_test_reduced)
      ```
      or
      ```
      from sklearn.feature_selection import RFE

      estimator = clone(best_clf)
      selector = RFE(estimator, n_features_to_select=5, step=1)
      selector = selector.fit(X_train, y_train)

      RFE_predictions = selector.predict(X_test)
      ```
  
### Final result
Accuracy on testing data: 0.8556, and 
F-score on testing data: 0.7408



