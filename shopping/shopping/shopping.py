import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    """
    months = {"Jan":0,"Feb":1,"Mar":2,"Apr":3,"May":4,"June":5,"Jul":6,"Aug":7,"Sep":8,"Oct":9,"Nov":10,"Dec":11}
    labels = []
    with open(filename, encoding="utf-8") as f:
        reader = csv.reader(f)
        reader = list(reader)
        reader = reader[1:]
        fl = [0,2,4,10,11,12,13,14,15,16]
        for row in range(len(reader)):
            reader[row][10] = months[reader[row][10]]
            reader[row][15] = 1 if reader[row] == 'Returning_Visitor' else 0
            reader[row][16] = 1 if reader[row] else 0
            labels.append(True if reader[row][-1] == "TRUE" else False)
            reader[row] = reader[row][:-1]
            reader[row] = list(map(float,reader[row]))
            for i in fl:
                reader[row][i] = int(reader[row][i])
                
    return (reader, labels)



def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence,labels)
    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).
    """
    sensitivity = 0
    pos_total = 0
    specificity = 0
    neg_total = 0
    for i in range(len(predictions)):
        if labels[i] == 1:
            if predictions[i] == 1:
                sensitivity +=1
            pos_total += 1
        elif labels[i] == 0:
            if predictions[i] == 0:
                specificity += 1
            neg_total += 1
    return ((sensitivity / pos_total),(specificity / neg_total))


if __name__ == "__main__":
    main()
