import numpy as np
import matplotlib.pyplot as plt 

import pandas as pd  
import seaborn as sns 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score


class HousePrice:

    def __init__(self):
        column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
        # setting column names
        self.df = pd.read_csv('housing.csv', names=column_names,header=None,delimiter='\s+')
        
        self.y_predict = None
        self.feature_col = None
        self.X_test_feat=None
        self.X_train_feat=None
    
    def validate(self):
        n = len(self.df)
        n_test = int(0.2 * n)
        n_train = n - n_test
       
        df_train = self.df.iloc[:n_train].copy()
        df_test = self.df.iloc[n_train:].copy()
       
        Y_train = df_train.MEDV.values
        Y_test = df_test.MEDV.values
        del df_train['MEDV']
        del df_test['MEDV']
        return df_train,df_test,Y_train,Y_test

    
    def linear_reg_model(self,df_train,df_test,Y_train,Y_test):
        # taken five models
        # on the basis of absolute high correlation value with the price column
        models=[['RM'],['LSTAT'],['RM','LSTAT'],['RM','LSTAT','PTRATIO'],['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']]
        i=1
        # seleting features
        min_rmse = None
        for features in models:
            X_train=df_train[features].copy()
            X_test=df_test[features].copy()
            lin_reg = LinearRegression()
            lin_reg.fit(X_train, Y_train)
            Y_train_pred = lin_reg.predict(X_train)
            rmse = (np.sqrt(mean_squared_error(Y_train, Y_train_pred)))
            r2 = r2_score(Y_train, Y_train_pred)
            print('Model: {}'.format(i),"\n")

            print("The model performance for training set:")
            print('RMSE is {}'.format(rmse))
            print('R2 score is {}'.format(r2),"\n")

            # model evaluation for testing set

            Y_test_pred = lin_reg.predict(X_test)
            # root mean square error of the model
            rmse = (np.sqrt(mean_squared_error(Y_test, Y_test_pred)))

            # r-squared score of the model
            r2 = r2_score(Y_test, Y_test_pred)

            print("The model performance for testing set:")
            print('RMSE is {}'.format(rmse))
            print('R2 score is {}'.format(r2))
 
            if min_rmse == None or rmse < min_rmse:
                min_rmse=rmse
                self.feature_col=features
                self.y_predict_result=Y_test_pred
                self.X_test_feat=df_test[features]
                self.X_train_feat=df_train[features]
                
            i=i+1
        
        # Best feature we got with minimum RMSE is for column 'LSTAT'
        plt.figure(figsize=(5, 5), dpi=80)
        plt.scatter(Y_test, self.y_predict_result)
        plt.title("Scatter Plot of actual and predicted value - Linear-Regression")
        plt.xlabel("Actual Value of MEDV")
        plt.ylabel(" Predicted Value")
        plt.show()
        
        # print(self.y_predict_result)
        # print(self.X_test_feat.values.flatten())
        plt.scatter(self.X_test_feat.values.flatten(),Y_test)
        plt.plot(self.X_test_feat.values.flatten(),self.y_predict_result,color= 'red')
        plt.title("Line of best model of - Linear-Regression")
        plt.xlabel("DataPoint")
        plt.ylabel(" Predicted Value")
        plt.show()
        
        
    
    def polynomial_reg_model(self,Y_train,Y_test):
        X_test=self.X_test_feat
        X_train=self.X_train_feat
        poly_features = PolynomialFeatures(degree=2)
        # transforms the existing features to higher degree features.
        X_train_poly = poly_features.fit_transform(X_train)
  
        # fit the transformed features to Linear Regression
        poly_model = LinearRegression()
        poly_model.fit(X_train_poly, Y_train)
  
        # predicting on training data-set
        y_train_predicted = poly_model.predict(X_train_poly)
  
        # predicting on test data-set
        y_test_predict = poly_model.predict(poly_features.fit_transform(X_test))
  
        # evaluating the model on training dataset
        rmse_train = np.sqrt(mean_squared_error(Y_train, y_train_predicted))
        r2_train = r2_score(Y_train, y_train_predicted)
  
        # evaluating the model on test dataset
        rmse_test = np.sqrt(mean_squared_error(Y_test, y_test_predict))
        r2_test = r2_score(Y_test, y_test_predict)
        #print("The model performance for the training set")
        #print("-------------------------------------------")
        #print("RMSE of training set is {}".format(rmse_train))
        #print("R2 score of training set is {}".format(r2_train))
  
        print("\n")
  
        print("The model performance for the test set on degree: 2\n")
        print("RMSE of test set is {}".format(rmse_test))
        print("R2 score of test set is {}".format(r2_test))
        
        plt.figure(figsize=(5, 5), dpi=80)
        plt.scatter(Y_test, y_test_predict)
        plt.title("Scatter Plot of actual and predicted value - Polynomial-Regression-degree-2")
        plt.xlabel("Actual Value of MEDV")
        plt.ylabel(" Predicted Value")
        plt.show()
        
        
        plt.scatter(X_test.values.flatten(),Y_test)
        plt.plot(X_test.values.flatten(),y_test_predict,color= 'red')
        plt.title("Curve of best model of - Polynomial-Regression-degree-2")
        plt.xlabel("DataPoint")
        plt.ylabel(" Predicted Value")
        plt.show()
        
        # Calculate prection values for degree=20
        
        poly_features_deg_20 = PolynomialFeatures(degree=20)
        # transforms the existing features to higher degree features.
        X_train_poly_deg_20 = poly_features_deg_20.fit_transform(X_train)
  
        # fit the transformed features to Linear Regression
        poly_model_deg_20 = LinearRegression()
        poly_model_deg_20.fit(X_train_poly_deg_20, Y_train)
  
        # predicting on training data-set
        y_train_predicted_deg_20 = poly_model_deg_20.predict(X_train_poly_deg_20)
  
        # predicting on test data-set
        y_test_predict_deg_20 = poly_model_deg_20.predict(poly_features_deg_20.fit_transform(X_test))
        
        # Plot curve for degree-20
        
        plt.figure(figsize=(5, 5), dpi=80)
        plt.scatter(Y_test, y_test_predict_deg_20)
        plt.title("Scatter Plot of actual and predicted value - Polynomial-Regression-degree-20")
        plt.xlabel("Actual Value of MEDV")
        plt.ylabel(" Predicted Value")
        plt.show()
        
        
        plt.scatter(X_test.values.flatten(),Y_test)
        plt.plot(X_test.values.flatten(),y_test_predict_deg_20,color= 'red')
        plt.title("Curve of best model of - Polynomial-Regression-degree-20")
        plt.xlabel("DataPoint")
        plt.ylabel(" Predicted Value")
        plt.show()
        
        
        
    def multiple_regression(self):
        
        df_train,df_test,Y_train,Y_test = self.validate()
        
        df_x_corr = df_train.copy()
        df_x_corr['MEDV'] = Y_train
        # print(df_x_corr.head())
        
        # getting correlation values 
        fig, ax = plt.subplots(figsize=(10,10))
        correlation_matrix = self.df.corr().round(2)
        sns.heatmap(data=correlation_matrix,annot=True,linewidths=.5, ax=ax)
        
        # taking 4 features with maximum correlation:
        
        cor_var = df_x_corr.corr().abs()['MEDV'].sort_values(ascending=False)[1:5]
        df_train_multi=pd.DataFrame()
        df_test_multi=pd.DataFrame()
        
        for i,v in cor_var.items():
            df_train_multi[i]=df_x_corr[i]
            df_test_multi[i]=df_test[i]
        
        lin_reg_multi = LinearRegression()
        lin_reg_multi.fit(df_train_multi, Y_train)
        Y_pred = lin_reg_multi.predict(df_test_multi)
       
        rmse = (np.sqrt(mean_squared_error(Y_test, Y_pred)))
        r2 = r2_score(Y_test, Y_pred)
        
        adj_r2 = 1 - ((1 - pow(r2, 2)) * (len(Y_train) - 1)/(len(Y_train)-df_train_multi.shape[1]-1))
        print("RMSE: ",rmse)
        print("R2: ",r2)
        print("Adj R2: ",adj_r2)
        

def main():
    house_price=HousePrice()
    df_train,df_test,Y_train,Y_test=house_price.validate()
    house_price.linear_reg_model(df_train,df_test,Y_train,Y_test)
    house_price.polynomial_reg_model(Y_train,Y_test)
    house_price.multiple_regression()
    
    

if __name__ == "__main__":
    main()

