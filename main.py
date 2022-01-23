from getPokec import MyDataset

d = MyDataset("pokec2")
print(d.graph.adj().to_dense())