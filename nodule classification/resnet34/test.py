import cPickle
f = open("./data/bin/data_batch_1","rb")
d = cPickle.load(f)
s = d["data"]
print len(s)
