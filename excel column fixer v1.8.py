# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 14:24:59 2017

@author: backesj
"""

"""
This program takes excel data and uses machine learning to predict the 
probability of a colummn's value being incorrect. 
"""

import pandas as pd

# read in data

df = pd.read_excel(r'H:\Python Webscraping\TN\pandas_simple.xlsx', sheetname = 'Veterinary Facility')


colList = []
for i in df:
    colList.append(list(df[i]))

# create feature vectors. 
# feature vectors are hand made and include
    # 1. Number of special characters
    # 2. Number of digits
    # 3. number of characters
    # 4. length of value
    # 5. Number of words in the value
    

# complile a list special characters found in the excel file
    
specCharList = []
for i in colList:
    for j in i:
        try:
            for k in j:
                None
        except TypeError:
            None
        else:
            for k in j:
                if k.isalpha() == False:
                    if k.isdigit() == False:
                        if k != ' ':
                            #if k != '.':
                                #if k != ',':
                            specCharList.append(k)   


specCharList = dedup(specCharList)    
        


firstCol = ''.join(list(df.columns)[2:][0])
lastCol = ''.join(list(df.columns)[2:][-1])

columns = list(df.columns)[2:]
length = len(columns)

# create training dataset

dfTrain = df.copy()
dfTest = df.copy()

for index, row in dfTrain.iterrows():
    if str(row[lastCol]) == 'nan':
        dfTrain.drop(index, inplace=True)

for index, row in dfTest.iterrows():
    if str(row[lastCol]) != 'nan':
        dfTest.drop(index, inplace=True)


    
        


dfTrain['class'] = "    "
dfTest['class'] = "    "



dfDictTrain = {}
for i, ivalue in enumerate(columns):
    tempDf = dfTrain[['%s'  % ivalue,'class']].copy()
    tempDf[ivalue].fillna('', inplace=True)
    tempDf['len'] = tempDf[ivalue].apply(lambda x: len(x) )
    tempDf['char'] = tempDf[ivalue].apply(lambda x: sum(c.isalpha() for c in x))
    tempDf['num'] = tempDf[ivalue].apply(lambda x: sum(c.isdigit() for c in x))
    tempDf['sym'] = tempDf[ivalue].apply(lambda x: sum(x.count(c) for c in specCharList ))
    tempDf['words'] = tempDf[ivalue].apply(lambda x: len([c for c in x.split(' ') if len(c) > 1]))
    #tempDf['space'] = tempDf[ivalue].apply(lambda x: sum(x.count(c) for c in ' ' ))
    #tempDf['alnum'] = tempDf[]
    tempDf['class'] = ivalue
    #tempDf['prior'] = i+1
    del tempDf[ivalue]
    dfDictTrain['%s' % ivalue] = tempDf
    
dfDictTest = {}
for i, ivalue in enumerate(columns):
    tempDf = dfTest[['%s'  % ivalue,'class']].copy()
    tempDf[ivalue].fillna('', inplace=True)
    tempDf['len'] = tempDf[ivalue].apply(lambda x: len(x) )
    tempDf['char'] = tempDf[ivalue].apply(lambda x: sum(c.isalpha() for c in x))
    tempDf['num'] = tempDf[ivalue].apply(lambda x: sum(c.isdigit() for c in x))
    tempDf['sym'] = tempDf[ivalue].apply(lambda x: sum(x.count(c) for c in specCharList ))
    tempDf['words'] = tempDf[ivalue].apply(lambda x: len([c for c in x.split(' ') if len(c) > 1]))
    #tempDf['space'] = tempDf[ivalue].apply(lambda x: sum(x.count(c) for c in ' ' ))
    #tempDf['alnum'] = tempDf[]
    tempDf['class'] = ivalue
    #tempDf['prior'] = i+1
    del tempDf[ivalue]
    dfDictTest['%s' % ivalue] = tempDf
'''
dictList = []
for i in dfDict:
    dictList.append("dfDict['"+i+"']")
dictList2 = str(dictList).split()    
frames = [dfDict]
'''
'''
frames = [dfDict['col2'], dfDict['col3'], dfDict['col4'], dfDict['col5'],
          dfDict['col6'], dfDict['col7'], dfDict['col8'], dfDict['col9']]
'''
datasetTrain = {}
for i in range(len(columns)-1):
    if i == 0:
        datasetTrain = dfDictTrain['col%s' % str(i+1)].append(dfDictTrain['col%s' % str(i+2)])
    else:
        datasetTrain = datasetTrain.append(dfDictTrain['col%s' % str(i+2)])

datasetTest = {}
for i in range(len(columns)-1):
    if i == 0:
        datasetTest = dfDictTest['col%s' % str(i+1)].append(dfDictTest['col%s' % str(i+2)])
    else:
        datasetTest = datasetTest.append(dfDictTest['col%s' % str(i+2)])


          #dfDict['col6'], dfDict['col7'], dfDict['col8'], dfDict['col9']]


print(datasetTrain.describe())

print(datasetTrain.groupby('class').size())

datasetTrain.plot(kind='box', subplots=True, layout=(2,3), sharex=False, sharey=False)
plt.show()

datasetTrain.hist()
plt.show()


# scatter plot matrix
scatter_matrix(dataset)
plt.show()

# Split-out validation dataset
array = datasetTrain.values
X = array[:,1:6]
Y = array[:,0]
validation_size = .2
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

array = datasetTest.values
X = array[:,1:6]
Y = array[:,0]
validation_size = .99
seed = 7
X_train2, X_validation2, Y_train2, Y_validation2 = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric

scoring = 'accuracy'


# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

import operator

modelsDict = {'LR': LogisticRegression(), 
'LDA': LinearDiscriminantAnalysis(),
'KNN': KNeighborsClassifier(),
'CART': DecisionTreeClassifier(),
'NB': GaussianNB(),
'SVM': SVC()}

# evaluate each model in turn
results = []
names = []
allMSG = []
msgDict = {}
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    msgDict['%s' %msg.split(' ')[0].strip(':')] = (msg.split(' ')[1])
    allMSG.append(msg.split(' ')[1])
    if name == 'SVM':
        bestModel = max(msgDict.items(), key = operator.itemgetter(1))[0] 
    
allMSG.index(max(allMSG))
# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

'''
# Make predictions on validation dataset
lr = LogisticRegression()
lr.fit(X_train, Y_train)
predictions = lr.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
'''

# Make predictions on validation dataset
currModel = modelsDict['%s' % bestModel]
currModel.fit(X_train, Y_train)
predictions = currModel.predict(X_validation2)
print(accuracy_score(Y_validation2, predictions))
print(confusion_matrix(Y_validation2, predictions))
print(classification_report(Y_validation2, predictions))



from sklearn import metrics
clf_rep = metrics.precision_recall_fscore_support(Y_validation2, predictions)
out_dict = {
             "precision" :clf_rep[0].round(2)
            ,"recall" : clf_rep[1].round(2)
            ,"f1-score" : clf_rep[2].round(2)
            ,"support" : clf_rep[3]
            }

df3 = pd.DataFrame.from_dict(out_dict, orient='Index')

df3 = df3.drop(['support'])


for i in range(len(columns)-1):
    df3['diff%s' % str(i+1)] = df3[i] - df3[i+1]
    
############ right here need to get columns of first 0s

df3 = df3.loc[:, 'diff1':'diff%s' % str(length-1)]



maxVal = df3.idxmax(axis=1)['f1-score']

# convert to original col name
badCol = maxVal.replace('diff','col').replace('%s' % maxVal[-1:], '%s' % str(int(maxVal[-1:])+1) )
joinCol = maxVal.replace('diff','col').replace('%s' % maxVal[-1:], '%s' % str(int(maxVal[-1:])) )
firstCol = ''.join(columns[0])
lastCol = ''.join(columns[-1:])

dfFirst = df.loc[:, '%s' % firstCol:'%s' % joinCol]
dfLast = df.loc[:, '%s' % origCol:'%s' % lastCol]


for index, row in dfLast.iterrows():
    if str(row[lastCol]) == 'nan':
        dfLast.loc[index] = dfLast.loc[index].shift(1)


result = pd.concat([dfFirst, dfLast], axis=1)







