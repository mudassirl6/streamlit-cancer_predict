import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error,accuracy_score,classification_report
import pickle as p



def get_clean_data():
    data = pd.read_csv("APP-CANCER/DATA/data.csv")

    data = data.drop(columns=['Unnamed: 32','id'],axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M':1,'B':0})
    return data


def create_model(data):
    X = data.drop(['diagnosis'],axis=1)
    y = data['diagnosis']
    
    #Scaling ...
    
    scaler = StandardScaler()
    
    X = scaler.fit_transform(X)
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=23)
    
    model = LogisticRegression()
    
    model.fit(X_train,y_train)
    
    #Test the model.....
    y_pred = model.predict(X_test)
    print("The mean squared error is .=  ",mean_squared_error(y_pred,y_test))
    print("The accuracy of the model is .=  ",accuracy_score(y_test,y_pred))
    print("The classification report is: \n",classification_report(y_test,y_pred))
    
    return model,scaler
    
    
# def test_model(model):
#     pass
    
    

def main():
    data = get_clean_data()

    model,scaler = create_model(data)

    with open('./model.pkl','wb') as f:
        p.dump(model,f)
    with open('./scaler.pkl', 'wb') as f:
        p.dump(scaler,f)
    

    

if __name__ == "__main__":
    main()