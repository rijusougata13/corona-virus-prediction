    #!/usr/bin/env python3
    # -*- coding: utf-8 -*-
    """
    Created on Thu Mar 19 19:42:51 2020
    
    @author: sougata
    """
   #importing lib 
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn import model_selection
    from sklearn.svm import SVR
    from sklearn.linear_model import LinearRegression
    import csv
    import datetime as dt
    from scipy.optimize import curve_fit
    from sklearn.preprocessing import PolynomialFeatures
    import numpy as np
    import xgboost
    from sklearn.ensemble import RandomForestRegressor 
    
   #importing dataset 
    df=pd.read_csv('time_series_covid_19_deaths.csv')
    
   #formating the raw data into good shape
    df['date'] = pd.to_datetime(df['date'])
    df['date']=df['date'].map(dt.datetime.toordinal)
    
   
    
    x=np.array(df['date'])
    x = x.reshape(-1, 1)
    y=np.array(df['death'])
    y=y.reshape(-1,1)
   
    
    x1=[]
    x1.append(input("input the date"))
    
    x1= pd.to_datetime(x1)
    x1=x1.map(dt.datetime.toordinal)
    x11=[x1]
   #spliting into train and test data 
    x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=0.2)
   
    
   
    #random forest regression
    
    regressor = RandomForestRegressor(n_estimators = 100, random_state = 0) 
    regressor.fit(x_train, y_train)  
    
  #  model1 = xgboost.XGBClassifier()
   # model1.fit(x_train,y_train)
    
   # print(model1.score(x_test,y_test))
    print("accuracy__>")
    print(regressor.score(x_test,y_test))
    
    plt.scatter(x_test, y_test,  color='black')
   # plt.plot(x_test,model1.predict(x_test), color = 'green')
   # plt.plot(x_test, pol_reg.predict(poly_reg.fit_transform(x_test)), color='green')
    plt.plot(x_test,regressor.predict(x_test), color = 'green')
    plt.scatter(x11,regressor.predict(x11),color='red')
    plt.title('corona prediction')
    plt.xlabel('date')
    plt.ylabel('corona_death')
    plt.show()
    ans=regressor.predict(x11)
    #print('total death -->")
    print(ans)
    
    #29-03-2020