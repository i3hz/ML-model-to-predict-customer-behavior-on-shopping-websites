import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python shopping.py data [k=1]")
    if len(sys.argv) == 3:
        try:
            k = int(sys.argv[2])
        except ValueError:
            sys.exit("k setting must be an integer")
    else:
        k = 1

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train, k)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    
    # Dictionary Mapping Months to Numerical values
    months = {'Jan': 0, 'Feb': 1, 'Mar': 2, 'Apr': 3, 'May': 4, 'June': 5, 'Jul': 6, 'Aug': 7, 'Sep': 8, 'Oct': 9, 'Nov': 10, 'Dec': 11}

    # Mapping Visitor Types to integers
    visitors = {'Returning_Visitor': 1, 'New_Visitor': 0, 'Other': 0}

    # Mapping Boolean Strings to integers
    bools = {'TRUE': 1, 'FALSE': 0}

    # Create list of lists for evidence, list for labels:
    evidence = []
    labels = []

    # Open CSV file and load in data as dict:
    with open(filename, newline='') as csvfile:
        csvreader = csv.DictReader(csvfile, delimiter=',')
        print('Loading Data from csv file...')
        lines = 0
        for row in csvreader:

            lines += 1

            line = []

            # Append Evidence to List of Lists
            line.append(int(row['Administrative']))
            line.append(float(row['Administrative_Duration']))
            line.append(int(row['Informational']))
            line.append(float(row['Informational_Duration']))
            line.append(int(row['ProductRelated']))
            line.append(float(row['ProductRelated_Duration']))
            line.append(float(row['BounceRates']))
            line.append(float(row['ExitRates']))
            line.append(float(row['PageValues']))
            line.append(float(row['SpecialDay']))
            line.append(months[row['Month']])
            line.append(int(row['OperatingSystems']))
            line.append(int(row['Browser']))
            line.append(int(row['Region']))
            line.append(int(row['TrafficType']))
            line.append(visitors[row['VisitorType']])
            line.append(bools[row['Weekend']])

            # Add evidence line to evidence
            evidence.append(line)

            # Append Labels to List
            labels.append(bools[row['Revenue']])

        # Confirm data loaded in successfully:
        if len(evidence) != len(labels):
            sys.exit('Error when loading data! Evidence length does not match label length')

        print('Data loaded successfully from csv file! Total lines: ', lines)

        return evidence, labels


def train_model(evidence, labels, k):
    

    print('Fitting Model using k-Nearest Neighbours Classifier, with k = ', k)

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(evidence, labels)

    return model


def evaluate(labels, predictions):
    
    pos_labels = labels.count(1)
    neg_labels = labels.count(0)

    correct_positive = 0
    correct_negative = 0

    for i in range(len(predictions)):
        if predictions[i] == labels[i]:
            if predictions[i] == 1:
                correct_positive += 1
            else:
                correct_negative += 1

    sensitivity = correct_positive / pos_labels
    specificity = correct_negative / neg_labels

    return sensitivity, specificity


if __name__ == "__main__":
    main()
