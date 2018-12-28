import os
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

activities = {"Brush_teeth": 0, "Climb_stairs": 1, "Comb_hair": 2,
              "Descend_stairs": 3, "Drink_glass": 4, "Eat_meat": 5,
              "Eat_soup": 6, "Getup_bed": 7, "Liedown_bed": 8,
              "Pour_water": 9, "Sitdown_chair": 10, "Standup_chair": 11,
              "Use_telephone": 12, "Walk": 13}

act = {0: "Brush_teeth", 1: "Climb_stairs", 2: "Comb_hair",
              3: "Descend_stairs", 4: "Drink_glass", 5: "Eat_meat",
              6: "Eat_soup", 7: "Getup_bed", 8: "Liedown_bed",
              9: "Pour_water", 10: "Sitdown_chair", 11: "Standup_chair",
              12: "Use_telephone", 13: "Walk"}


def flatten(list, fd):
    flatten = []
    flatten.append(activities[fd])
    for x in list:
        for y in x.split(" "):
            flatten.append(y)
    return flatten

def parse():
    data = []
    files = os.listdir("Data")
    train = []
    test = []
    for fd in files:
        signals = os.listdir("Data/%s" %fd)
        fn = "Data/%s/" % fd
        for sig in signals:
            with open(fn+"%s" % sig) as f:
                lines = f.read().splitlines()
                signal = flatten(lines, fd)
                data.append(np.asarray(signal))
    while len(set([x[0] for x in data[:100]])) != 14:  # make sure all the types of activities appear in the test data
        random.shuffle(data)
        test = data[:100]
        train = data[100:]

    return train, test


def makeSegments(data, seg_length):
    segments = []
    segment_file_no = []
    for idx, file in enumerate(data):
        segNumber = len(file) // seg_length
        temp = file.copy()
        for i in range(segNumber):
            segment_file_no.append(idx)
            segments.append(temp[:seg_length])
            temp = temp[seg_length:]
    return segments, segment_file_no


def feature_vec(fv, segments, seg_file_no, labels):
    for i in range(len(segments)):
        fv[seg_file_no[i]][labels[i]] += 1

if __name__ == "__main__":
  for m in range(5):                                # run each case 5 times
    K = 50
    seg_length = 3 * 32
    train, test = parse()
    train_labels = [x[0] for x in train]
    train_data = np.array([x[1:] for x in train])
    test_labels = [x[0] for x in test]
    test_data = np.array([x[1:] for x in test])

    # making segments
    train_segments, train_seg_file_no = makeSegments(train_data, seg_length)
    test_segments, test_seg_file_no = makeSegments(test_data, seg_length)

    feature_vectors = [[0 for col in range(K)] for row in range(len(train))]
    kmeans = KMeans(n_clusters=K).fit(train_segments)

    feature_vec(feature_vectors, train_segments, train_seg_file_no, kmeans.labels_)

    print(len(feature_vectors[0]))

    # making histogram for each activity
    for label in range(14):
        indices = [i for i, x in enumerate(train_labels) if label == int(x)]
        sum = np.array([0 for col in range(K)])

        for idx in indices:
            sum += np.array(feature_vectors[idx])
        avg = sum/len(train_labels)
        plt.figure(0)
        plt.bar(range(K), avg)
        plt.title(act[label])
        plt.show()

    tkmeans = kmeans.predict(test_segments)

    t_feature_vectors = [[0 for col in range(K)] for row in range(len(test))]
    feature_vec(t_feature_vectors, test_segments, test_seg_file_no, tkmeans)

    # classifier
    clf = RandomForestClassifier(n_estimators=200, max_depth=6)
    clf.fit(feature_vectors, train_labels)
    testing_results = clf.predict(t_feature_vectors)

    accuracy = np.sum(testing_results == test_labels) / len(test)
    error_rate = 1-accuracy
    print(error_rate, accuracy)

    c_mat = confusion_matrix(test_labels, testing_results)
    print(c_mat)

    m += 1
