import csv
import random
import math
import operator


def loadDataset(filename, split, trainingSet=[], testSet=[]):
    with open(filename, 'rb') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)-1):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
            #if random.randrange(len(trainingSet)) < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])


def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x]-instance2[x]), 2)
    return math.sqrt(distance)


def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
        return neighbors


def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        pass
        # print 'test'
        # test = testSet[x][-1]
        # print test
        # print 'pre'
        # pre = predictions[x]
        # print pre
    print ('test: ' + repr(testSet[x][-1]))
    print ('pre: ' + repr(predictions[x]))
    # if testSet[z][-1] == predictions[z]:
    #     correct += 1
    return (correct/float(len(testSet)))*100.0


def main():
    #prepare data
    """

    :rtype: object
    """
    trainingSet = []
    testSet = []
    split = 0.70
    loadDataset(r'/home/zhoumiao/ML/02KNearestNeighbor/irisdata.txt', split, trainingSet, testSet)
    print('Train set: ' + repr(len(trainingSet)))
    print('Test set: ' + repr(len(testSet)))
    #generate predictions
    predictions = []
    k = 3
    correct = []
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        #print ('test: ' + repr(testSet))
        print ('predictions: ' + repr(predictions))
        print ('>predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))

        if result == testSet[x][-1]:
            correct.append(x)
            # print "len:"
            # print len(testSet)
            # print "correct:"
            # print len(correct)
    accuracy = (len(correct)/float(len(testSet)))*100.0
    print('Accuracy: ' + repr(accuracy) + '%')


if __name__ == '__main__':
    main()




























