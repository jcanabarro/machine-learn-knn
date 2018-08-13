import numpy as np


class KNN:

    def train(self, X, Y):
        self.X = X
        self.Y = Y

    # neighbor_mode:
    #   0 -> Inverso da distancia euclidiana
    #   1 -> 1 - Distancia normalizada
    # vote_mode:
    #   0 -> voto majoritario
    #   1 -> voto ponderado
    def predict(self, X, Y, k=1, neighbor_mode=0, vote_mode=0):
        instance_votes = []
        for test_index, test_instance in X.iterrows():
            test = np.array(test_instance)
            dists = []
            for train_index, train_instance in self.X.iterrows():
                d = np.linalg.norm(test - np.array(train_instance))
                dists.append(d)
            dists = np.array(dists)
            if neighbor_mode == 1:
                dists = (dists - np.min(dists)) / (np.max(dists) - np.min(dists))
            dists2 = []
            for index, d in enumerate(dists):
                if neighbor_mode == 0:
                    dists2.append((1 / d, index))
                elif neighbor_mode == 1:
                    dists2.append((1 - d, index))
            instance_votes.append(sorted(dists2, reverse=True)[:k])
        predictions = []
        for votes in instance_votes:
            votes_count = {}
            for vote in votes:
                v = self.Y.iloc[vote[1]][0]
                if not v in votes_count:
                    if vote_mode == 0:
                        votes_count[v] = 1
                    elif vote_mode == 1:
                        votes_count[v] = 1 / vote[0]
                else:
                    if vote_mode == 0:
                        votes_count[v] += 1
                    elif vote_mode == 1:
                        votes_count[v] += 1 / vote[0]
            predictions.append(sorted(votes_count.items(), reverse=True)[0])

        acc = 0

        for index, p in enumerate(predictions):
            expected = Y.iloc[index][0]
            acc += expected == p[0]

        return acc / len(X)
