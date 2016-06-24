from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np


X = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])

y = np.array([[0],
			[1],
			[1],
			[0]])

lr = LogisticRegression()
lr.fit(X,y)
print accuracy_score(lr.predict(X),y)
