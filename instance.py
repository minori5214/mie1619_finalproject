import tsplib95
import tsplib95.distances as distances

#def euclidean(start, end)

class Instance_MIP():
    def __init__(self, path):
        self.data = tsplib95.load(path)
        self.N = self.data.dimension

        self.dist = []
        if self.data.edge_weight_type == 'EXPLICIT' and \
            self.data.edge_weight_format in ['LOWER_DIAG_ROW', 'UPPER_ROW', 'UPPER_DIAG_ROW']:
            for i in range(self.N):
                self.dist.append([self.data.get_weight(i, j) for j in range(self.N)])

        elif self.data.edge_weight_type == 'GEO':
            for i in range(1, self.N+1):
                row = []
                for j in range(1, self.N+1):
                    #distance = distances.euclidean(self.data.node_coords[i], self.data.node_coords[j], lambda x: x)

                    # Make distance all integer for CP
                    distance = distances.euclidean(self.data.node_coords[i], self.data.node_coords[j])
                    row.append(distance)
                self.dist.append(row)

        elif self.data.edge_weight_type in ['EUC_2D', 'ATT', 'CEIL_2D']:
            cities = [(city, tuple(coord)) for city, coord in self.data.node_coords.items()]
            for i in range(self.N):
                self.dist.append([distances.euclidean(cities[i][1], node[1]) for node in cities])

        elif self.data.edge_weight_type == 'EXPLICIT' and \
            self.data.edge_weight_format == 'FULL_MATRIX': 
            self.dist = self.data.edge_weights
        
        else:
            raise NotImplementedError

class Instance_CP():
    def __init__(self, path):
        self.data = tsplib95.load(path)
        self.N = self.data.dimension

        self.dist = []
        if self.data.edge_weight_type == 'EXPLICIT' and \
            self.data.edge_weight_format in ['LOWER_DIAG_ROW', 'UPPER_ROW', 'UPPER_DIAG_ROW']:
            for i in range(self.N):
                tmp = [self.data.get_weight(i, j) for j in range(self.N)]
                tmp.append(self.data.get_weight(i, 0))
                self.dist.append(tmp)
            tmp = [self.data.get_weight(0, j) for j in range(self.N)]
            tmp.append(self.data.get_weight(0, 0))
            self.dist.append(tmp)

        elif self.data.edge_weight_type == 'GEO':
            for i in range(1, self.N+1):
                row = []
                for j in range(1, self.N+1):
                    distance = distances.euclidean(self.data.node_coords[i], self.data.node_coords[j])
                    row.append(distance)
                row.append(distances.euclidean(self.data.node_coords[i], self.data.node_coords[1]))
                self.dist.append(row)
            row = []
            for j in range(1, self.N+1):
                distance = distances.euclidean(self.data.node_coords[1], self.data.node_coords[j])
                row.append(distance)
            row.append(distances.euclidean(self.data.node_coords[1], self.data.node_coords[1]))
            self.dist.append(row)

        elif self.data.edge_weight_type in ['EUC_2D', 'ATT', 'CEIL_2D']:
            cities = [(city, tuple(coord)) for city, coord in self.data.node_coords.items()]
            for i in range(self.N):
                tmp = [distances.euclidean(cities[i][1], node[1]) for node in cities]
                tmp.append(distances.euclidean(cities[i][1], cities[0][1]))
                self.dist.append(tmp)
            tmp = [distances.euclidean(cities[0][1], node[1]) for node in cities]
            tmp.append(distances.euclidean(cities[0][1], cities[0][1]))
            self.dist.append(tmp)

        elif self.data.edge_weight_type == 'EXPLICIT' and \
            self.data.edge_weight_format == 'FULL_MATRIX': 

            for i in range(self.N):
                tmp = self.data.edge_weights[i]
                tmp.append(self.data.edge_weights[i][0])
                self.dist.append(tmp)
            
            self.dist.append(self.data.edge_weights[0])

    def optimal_tour(self, opt_path):
        self.opt_path = opt_path
        solution = tsplib95.load(self.opt_path)
        tour = [[x-1 for x in solution.tours[0]]]
        print("Optimal tour: ", tour[0])

        sum = []
        for i in range(self.N-1):
            sum.append(self.dist[tour[0][i]][tour[0][i+1]])
        sum.append(self.dist[tour[0][-1]][tour[0][0]])
        print(sum)
        print('Total optimal tour length is {0}.'.format(sum))

if __name__ == "__main__":
    instance = Instance_MIP('./instances/ulysses16.tsp')