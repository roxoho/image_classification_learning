import os
import pickle
from skimage.io import imread
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#prepare data
input_dir = './parking-data/'
categories = ['empty', 'occupied']

data =[]
labels = []

for idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir, category)):
        img_path = os.path.join(input_dir, category, file)
        img = imread(img_path)
        img = resize(img, (15,15))
        data.append(img.flatten())
        labels.append(idx)

data = np.asarray(data)
labels = np.asarray(labels)


#tran/test split
x_train,x_test,y_train,y_test = train_test_split(data,labels,test_size=0.2,stratify=labels,shuffle=True)


#train classifier
classifier = SVC()

parameters = [{'gamma': [0.01,0.001,0.0001],'C': [1,10,100,1000]}]

grid_search = GridSearchCV(classifier,parameters)

grid_search.fit(x_train,y_train)




#test performance

best_estimator = grid_search.best_estimator_
y_predict = best_estimator.predict(x_test)
accuracy = accuracy_score(y_predict,y_test)

print('{}% of samples were correctly classified'.format(str(accuracy*100)))

pickle.dump(best_estimator,open('classifier.p','wb'))