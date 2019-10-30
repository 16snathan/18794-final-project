import numpy as np
from numpy import genfromtxt
from sklearn.metrics import confusion_matrix

csv = genfromtxt('testing_res.csv',delimiter=',',dtype=None,encoding="utf8")


dat = csv[1:,:]



letters = np.array(['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y'])

#function outputs real class and the predicted class for one row of the csv
def act_pred_let( x ):
    return np.array([x[0], letters[np.argmax((x[1:]).astype(float))]])


#runs the function for every row
res = np.apply_along_axis(act_pred_let, axis=1,arr = dat)


con_mat = np.array(confusion_matrix(res[:,0], res[:,1]))


np.savetxt('confusion_mat.csv',con_mat,delimiter = ',', fmt= '%d')

print('Total Accuracy = ',end='')
print(((res[:,0] == res[:,1]).sum())/(res.shape[0]))
