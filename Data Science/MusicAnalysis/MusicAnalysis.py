from pandas import read_csv
from random import shuffle

from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.utils import shuffle


class MusicAnalysis:
    def main(self):
        df = read_csv("music.csv")

        '''
        Important: Before splitting the data for training and test sets,
        if just assigning the first X rows for testing (e.g., df.iloc[:500,:])
        then some classes might completely disappear from the dataset. One solution
        to that is to shuffle the rows in the CSV file such that the samples from the
        different classes appear in random order. In that case separating the first X
        rows will assign random samples for training and testing. Another solution is
        to use the “shuffle” function, that shuffles the rows in the data frame.
        '''
        df = shuffle(df)

        # Same as in Weka, drop 'Path' column
        df = df.drop(['Path'],axis=1)

        ten_percent_of_data = round(len(df) * 0.10)  # Approx 288 out of 2879 samples

        # Split the data into training and test sets
        train_set = df.iloc[:ten_percent_of_data, :]  # Assign to first 288 rows of df
        test_set = df.iloc[ten_percent_of_data:, :]  # Assign to last rows of df

        # Scikit-learn requires list of lables as input
        train_labels = train_set["Class"].tolist()
        # Remove class column from train sets
        train_set = train_set.drop(["Class"], axis=1)
        # Convert dataframe to list of lists
        train_samples = train_set.values.tolist()

        # Scikit-learn requires list of labels as input
        test_labels = test_set["Class"].tolist()
        # Remove class column from tests sets
        test_set = test_set.drop(["Class"], axis=1)
        # Convert dataframe to list of lists
        test_samples = test_set.values.tolist()

        classifiers = [
            KNeighborsClassifier(3),
            SVC(kernel="rbf", C=0.025, probability=True),
            NuSVC(nu=0.1,probability=True),
            DecisionTreeClassifier(),
            RandomForestClassifier(),
            AdaBoostClassifier(),
            GradientBoostingClassifier(),
            GaussianNB(),
            LinearDiscriminantAnalysis(),
            QuadraticDiscriminantAnalysis()
        ]

        for classifier in classifiers:
            classifier.fit(train_samples, train_labels)
            result = classifier.predict(test_samples)

            acc = accuracy_score(test_labels, result)
            print("{} Accuracy: {}".format(classifier.__class__.__name__, acc))

if __name__ == "__main__":
    analysis = MusicAnalysis()
    analysis.main()
