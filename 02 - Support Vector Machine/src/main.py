from SVM import * #import semua yang ada di file SVM.py

x = [[0], [1], [2], [3]]
y = [0, 1, 2, 3]
clf = svm.SVC(decision_function_shape='ovo')
clf.fit(x, y)
dec = clf.decision_function([[1]])
dec.shape[1] # 4 classes: 4*3/2 = 6
clf.decision_function_shape = "ovr"
dec = clf.decision_function([[1]])
dec.shape[1] # 4 classes