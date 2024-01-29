import pandas as pd
import requests
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split

# Carregar o conjunto de dados de treino
dataset = pd.read_csv('abalone_dataset.csv')
dataset['sex'] = dataset['sex'].map({'M': 0, 'F': 1, 'I': 2})
X = dataset.drop('type', axis=1)
y = dataset['type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

clf = SVC(kernel='linear', C=1).fit(X_train, y_train)

# Treinando o modelo
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print('Accuracy with cross validation', accuracy_score(y_test, y_pred))

X_validation = pd.read_csv('abalone_app.csv')
X_validation['sex'] = X_validation['sex'].map({'M': 0, 'F': 1, 'I': 2})

predictions = clf.predict(X_validation)

print(predictions)

URL = "https://aydanomachado.com/mlclass/03_Validation.php"
DEV_KEY = "Automata"

# json para ser enviado para o servidor
data = {'dev_key':DEV_KEY,
        'predictions':pd.Series(predictions).to_json(orient='values')}

# Enviando requisição e salvando o objeto resposta
r = requests.post(url = URL, data = data)

# Extraindo e imprimindo o texto da resposta
pastebin_url = r.text
print(" - Resposta do servidor:\n", r.text, "\n")