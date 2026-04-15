
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

def Advertise(Datapath):
    Border = "-"*40
    #---------------------------
    # STEP 1 : LOAD DATASET
    #---------------------------

    print(Border)
    print("STEP 1 : LOAD DATASET")
    print(Border)

    df = pd.read_csv(Datapath)

    print("FEW RECORDS FROM THE DATASET")
    print(df.head)

    #---------------------------
    # STEP 2 : REMOVE UNWANTED COLUMNS
    #---------------------------
    print(Border)
    print("STEP 2 : REMOVE UNWANTED COLUMNS")
    print(Border)

    print("SHAPE OF DATASET BEFORE REMOVAL :",df.shape)

    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'],inplace=True)

    print("SHAPE OF DATASET AFTER REMOVAL :",df.shape)    

    print(Border)
    print("CLEAN DATASET IS :")
    print(Border)

    print(df.head())

    #---------------------------
    # STEP 3 : CHECK MISSING VALUES
    #---------------------------
    print(Border)
    print("STEP 3 : CHECK MISSING VALUES")
    print(Border)

    print("MISSING VALUES COUNT :\n",df.isnull().sum())

    #---------------------------
    # STEP 4 : DISPLAY STATISTICAL SUMMARY
    #---------------------------
    print(Border)
    print("STEP 4 : DISPLAY STATISTICAL SUMMARY")
    print(Border)

    print(df.describe())

    #---------------------------
    # STEP 5 : CORRELATION BETWEEN COLUMNS 
    #---------------------------
    print(Border)
    print("STEP 5 : CORRELATION BETWEEN COLUMNS")
    print(Border)

    print("CORRELATION MATRIX")
    print(df.corr())

    #---------------------------
    # STEP 6 : SPLIT DATASET INTO DEPENDENT AND INDEPENDENT VARIABLES
    #---------------------------
    print(Border)
    print("STEP 6 : SPLIT DATASET INTO DEPENDENT AND INDEPENDENT VARIABLES")
    print(Border)

    X = df[['TV','radio','newspaper']]
    Y = df['sales']

    print("SHAPE OF INDEPENDENT VARIABLES :",X.shape)
    print("SHAPE OF DEPENDENT VARIABLES :",Y.shape)

    #---------------------------
    # STEP 7 : SPLIT DATASET FOR TRAINING AND TESTING
    #---------------------------
    print(Border)
    print("STEP 7 : SPLIT DATASET FOR TRAINING AND TESTING")
    print(Border)

    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
    print("X_TRAIN SHAPE :",X_train.shape)
    print("X_TEST SHAPE :",X_test.shape)
    print("Y_TRAIN SHAPE :",Y_train.shape)
    print("Y_TEST SHAPE :",Y_test.shape)

    #---------------------------
    # STEP 8 : CREATE & TRAIN THE MODEL
    #---------------------------
    print(Border)
    print("STEP 8 : CREATE & TRAIN THE MODEL")
    print(Border)

    model = LinearRegression()

    model.fit(X_train,Y_train)

    #---------------------------
    # STEP 9 : TEST THE MODEL
    #---------------------------
    print(Border)
    print("STEP 9 : TEST THE MODEL")
    print(Border)

    Y_pred = model.predict(X_test)

    #---------------------------
    # STEP 10 : EVALUATE THE MODEL
    #---------------------------
    print(Border)
    print("STEP 10 : EVALUATE THE MODEL")
    print(Border)

    MSE = mean_squared_error(Y_test,Y_pred)

    RMSE = np.sqrt(MSE)

    R2 = r2_score(Y_test,Y_pred)

    print("MEAN SQUARE ERROR :",MSE)

    print("ROOT MEAN SQUARED ERROR :",RMSE)

    print("R SQUARE VALUE :",R2)

    #---------------------------
    # STEP 11 : CALCULATE MODEL COEFFICIENT
    #---------------------------
    print(Border)
    print("STEP 11 : CALCULATE MODEL COEFFICIENT")
    print(Border)

    for column, value in zip(X.columns, model.coef_):
        print(f" {column} : {value} ")

    print("INTERCEPT :",model.intercept_)

    #---------------------------
    # STEP 12 : COMPARE THE ACTUAL AND PREDICTED VALUES
    #---------------------------
    print(Border)
    print("STEP 12 : COMPARE THE ACTUAL AND PREDICTED VALUES")
    print(Border)

    Result = pd.DataFrame({
        'ACTUAL SALE' : Y_test.values,
        'PREDICTED SALE' : Y_pred
        })

    print(Result.head())

    #---------------------------
    # STEP 13 : PLOT ACTUAL VS PREDICTED
    #---------------------------
    print(Border)
    print("STEP 13 : PLOT ACTUAL VS PREDICTED")
    print(Border)

    plt.figure(figsize=(8,5))
    plt.scatter(Y_test,Y_pred)
    plt.xlabel("ACTUAL SALES")
    plt.ylabel("PREDICYED SALES")
    plt.title("ACTUAL SALES VS PREDICTED SALES")
    plt.grid(True)
    plt.show()

def main():
    Advertise("Advertising.csv")

if __name__ == "__main__":
    main()