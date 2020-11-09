import pandas as pd
from math import exp,sqrt,pi

""" Naive Bayes Classifier : Class"""
class Bayes: 
    def __init__(self,data,labelColumn,predictors,ratio):
        self.data=data
        print(self.data.head())
        boundary=int(self.data.shape[0]*ratio)
        self.test=data.iloc[:boundary]
        self.train=data[boundary:]
        self.labelColumn=labelColumn
        self.labels=list(set(self.test[labelColumn].values.tolist()))
        self.predictors=predictors
        
    """ 
        Compute the probability of each class
        P(True) & P(False)
    """
    def ProbaLabel(self):
        total=self.test.shape[0]
        count=[]
        for label in self.labels:
            byLabel=self.test[self.test[self.labelColumn]==label]
            count.append(byLabel.shape[0]/total)
        return count
    
    """
        Compute the mean & standard deviation for each predictor given a certain class
        ex : P('accousticness'|True)
    """
    def RawConditionnelle(self):
        res=[]
        #gaussian=exp(-((x-mean)**2 / (2 * stdev**2 )))
        for label in self.labels:
            temp=[]
            byLabel=self.test[self.test[self.labelColumn]==label]
            for predictor in self.predictors:
                serie=byLabel[predictor]
                temp.append((serie.mean(),serie.std(),serie.shape[0]))
            res.append(temp)
        return res
    
    """ Compute the Gaussian Function for probability"""
    def ComputeGaussian(self,x,mean,std):
        exponent = exp(-((x-mean)**2 / (2 * std**2 )))
        return (1 / (sqrt(2 * pi) * std)) * exponent
    
    """Compute the probability of an ind of belonging to the different class """
    def GaussianProba(self,x):
        raw=self.RawConditionnelle()
        probaLabel=self.ProbaLabel()
        proba={}
        for i,label in enumerate(self.labels):
            data=raw[i]
            proba[label]=probaLabel[i]
            for j in range(len(self.predictors)):
                proba[label]*=self.ComputeGaussian(x[j],data[j][0],data[j][1])
        return proba

    def Predict(self,x):
        probabilites = self.GaussianProba(x)
        bestLabel,bestProb=None,-1
        for classValue, prob in probabilites.items():
            if bestLabel is None or prob>bestProb:
                bestProb=prob
                bestLabel=classValue
        return bestLabel
    
    def Accuracy(self):
        xTest=self.test[self.predictors].to_numpy()
        yTest=self.test[self.labelColumn].to_numpy()
        yPredict=[]
        for x in xTest:
            yPredict.append(self.Predict(x))
        correctPos,correctNeg,uncorrectPos,uncorrectNeg=0,0,0,0
        for i in range(len(yTest)):
            if yTest[i]==yPredict[i] and yPredict[i]==True:
                correctPos+=1
            elif yTest[i]==yPredict[i] and yPredict[i]==False:
                correctNeg+=1
            elif yTest[i]==True and yPredict[i]==False:
                uncorrectNeg+=1
            else: 
                uncorrectPos+=1
        print('Matrice de confusion :')
        print(' ','|',' 0  ','|',' 1 ')
        print('0','|',correctNeg/float(len(yTest)),'|',uncorrectNeg/float(len(yTest)))
        print('1','|',uncorrectPos/float(len(yTest)),'|',correctPos/float(len(yTest)))
        print()
        print('Total Accuracy :', (correctNeg+correctPos)/float(len(yTest)))
    

data=pd.read_csv('C:/Users/julie/Documents/01.Cours/01_ESILV/A4/PTS/PTS/data2017.csv')
data=data.drop(columns=["Unnamed: 0"])
data=data[data.year>2010]
print(data.head())
predictors=['artistPop','acousticness','danceability','duration_ms','energy','instrumentalness','liveness','loudness','speechiness','tempo','valence','year']
bayes=Bayes(data,'top',predictors,0.45)
print(bayes.RawConditionnelle())
#print(bayes.test)
testDF=bayes.test[predictors]
tests=testDF.to_numpy()
print(bayes.test.head(10))
for test in tests[:10]: 
     print(bayes.Predict(test))
bayes.Accuracy()
