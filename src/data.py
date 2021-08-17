import os, inspect, random, pickle
import numpy as np, scipy.sparse as sp
from tqdm import tqdm
import pickle


def load(args):
    """
    parses the dataset
    """
    dataset = parser(args.data, args.dataset).load_data() #실험 종류, 실험 데이터 parser함수 dict형태

    current = os.path.abspath(inspect.getfile(inspect.currentframe())) #data.py 파일 위치
    srcDir = os.path.dirname(current) #src파일 위치
    projectDir = os.path.dirname(srcDir) #project파일 위치
    file = os.path.join(projectDir, "datasets", args.data, args.dataset, "splits", str(args.split) + ".pickle") #pickle 위치

    if not os.path.isfile(file): print("split + ", str(args.split), "does not exist")   #해당 파일 없을시 출력
    with open(file, 'rb') as H: #file -> H pickle 파일 읽음
        Splits = pickle.load(H) #Splits에 pickle file load
        train, test = Splits['train'], Splits['test']  # key가 train 인 부분, test인 부분을 각각 나누어 할당, test,train value: [논문] 리스트 형식

    return dataset, train, test


class parser(object):
    """
    an object for parsing data
    """

    def __init__(self, data, dataset):
        """
        initialises the data directory

        arguments:
        data: coauthorship/cocitation
        dataset: cora/dblp/acm for coauthorship and cora/citeseer/pubmed for cocitation
        """

        current = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) #data.py 파일위치
        projectDir = os.path.dirname(current) #project 파일 위치
        self.d = os.path.join(projectDir, "datasets", data, dataset)  # d 변수 -> dataset 경로
        self.data, self.dataset = data, dataset #parser class에 data,dataset 변수대입

    def load_data(self):
        """
        loads the coauthorship hypergraph, features, and labels of cora

        assumes the following files to be present in the dataset directory:
        hypergraph.pickle: coauthorship hypergraph
        features.pickle: bag of word features
        labels.pickle: labels of papers

        n: number of hypernodes
        returns: a dictionary with hypergraph, features, and labels as keys
        """

        with open(os.path.join(self.d, 'hypergraph.pickle'), 'rb') as handle: #data경로에서 hypergraph 파일 read
            hypergraph = pickle.load(handle) #hypergraph 변수에 load hypergrpah {저자:[논문]}형식
            print("number of hyperedges is", len(hypergraph))

        with open(os.path.join(self.d, 'features.pickle'), 'rb') as handle:
            features = pickle.load(handle).todense() #feature 행렬dense화 numpy matrix

        with open(os.path.join(self.d, 'labels.pickle'), 'rb') as handle:
            labels = self._1hot(pickle.load(handle)) #label 1_hot encoding

        return {'hypergraph': hypergraph, 'features': features, 'labels': labels, 'n': features.shape[0]}

    def _1hot(self, labels):
        """
        converts each positive integer (representing a unique class) into ints one-hot form

        Arguments:
        labels: a list of positive integers with each integer representing a unique label
        """

        classes = set(labels)
        onehot = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
        return np.array(list(map(onehot.get, labels)), dtype=np.int32)

'''
    def parse(self):
        """
        returns a dataset specific function to parse
        """

        name = "_load_data"
        function = getattr(self, name, lambda: {})
        return function() #load data return
'''