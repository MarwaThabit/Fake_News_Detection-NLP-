import pickle 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)     
 
def plot_confusion_matrix(conf_matrix):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True,fmt='d',
                xticklabels=['Predicted_REAL', 'Predicted_FAKE'], 
                yticklabels=['Actual_REAL', 'Actual_FAKE'])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')
    plt.show()
    
def plot_accuracy(score):
    plt.figure(figsize=(6, 4))
    sns.barplot(x=['Accuracy'], y=[score])
    plt.ylim(0, 1)
    plt.title('Accuracy of the Model')
    plt.ylabel('Accuracy')
    plt.show()

def plot_label_distribution(y_pred):
    plt.figure(figsize=(6, 4))
    sns.countplot(x=y_pred)
    plt.title('Distribution of Predicted Labels')
    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.show()
    
def evaluate_model(score,y_test,y_pred):
    print("="*55)
    print(f'               Testing_Accuracy: {round(score * 100, 2)}%')

    conf_matrix = confusion_matrix(y_test, y_pred, labels=[428, 427])


    print("="*55)
    print("            Predicted_REAL  |   Predicted_FAKE  ")
    print("="*55)
    print(f" Actual_REAL   {conf_matrix[0,0]} (TP)     |     {conf_matrix[0,1]} (FN)")
    print(f" Actual_FAKE   {conf_matrix[1,0]}  (FP)    |    {conf_matrix[1,1]} (TN) ")
    
    plot_confusion_matrix(conf_matrix)
    
    plot_accuracy(score)
    
    plot_label_distribution(y_pred)  


def main():
    pac_model = load_model('PassiveAggressiveClassifier.pkl') 
    
    with open('y_test.pkl', 'rb') as f:
        y_test = pickle.load(f)
        
    with open('tfidf_test.pkl', 'rb') as f:
        tfidf_test = pickle.load(f) 
          
    y_pred =  pac_model.predict(tfidf_test)
    score = accuracy_score(y_test, y_pred)
    
    evaluate_model(score,y_test,y_pred)
    
main()
#4s