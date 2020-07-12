class Model(object):
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.N = x.shape[1]

    def fit(self,beta = 0.9, ler_rate = 0.01, epoch = 10000):
        self.theta = np.zeros((1, self.N)) # stworzenie wektora parametrów theta
        self.b = 0                         # wyraz wolny
        iteratrion = list(range(self.N))   # liczba iteracji 
        for i in range(epoch):
            
            grad_theta = np.zeros((1,self.N))  # wektor gradientów
            grad_b = 0                         # gradient wyrazu wolnego
            random.seed(i)                     # ustawienie tego samego ziarna dla każdej iteracji
            random.shuffle(iteratrion)         # przemieszanie danych 
            for j in iteratrion:
                prediction = np.dot(self.theta,x[j]) + self.b     # predykcja
                grad_theta = beta*grad_theta + ler_rate*((self.sigmoid(prediction) - y[j])*x[j]) #wykorzystanie poprzedniego gradnienu w celu przyspieszenia zbieżności (momentum)
                grad_b = beta*grad_b + ler_rate*(self.sigmoid(prediction) - y[j])

                self.theta = self.theta - grad_theta  #aktualizacja parametrów
                self.b = self.b - grad_b

    def sigmoid(self,x):      #funckja sigmoidalna, zapisana w inny sposób, aby uniknać problemów numerycznych
        if x >=0:
            z = np.exp(-x)
            return 1/(1 + z)
        else:
            z = np.exp(x)
            return z/(1 + z)




    def predict(self, X_test):                      #predykcja
        N = X_test.shape[0]
        self.predictions = np.zeros((1, N))

        for i in range(N):
            if self.sigmoid(np.dot(self.theta, X_test[i]) + self.b) > 0.5:  #klasyfikacja do odpowiedniej cechy na postawie wyniku funckji sigmoidalnej
                self.predictions[0][i] = 1
                
    @staticmethod
    def evaluate(Y_test, predictions):  #wyliczenie accuraccy
        N = Y_test.shape[0]
        correct = 0
        for i in range(N):
            if predictions[0][i] == Y_test[i]: 
                correct += 1
        accuracy =  correct / N
        return accuracy
    
    
    @staticmethod
    def confusion_matrix(Y_test, predictions):  #wyznaczenie macierzy błędu
        N = Y_test.shape[0]
        conf_matrix = np.zeros((2,2))

        for i in range(N):
            if predictions[0][i] == Y_test[i]:
                index = int(predictions[0][i])  #zakwalifikowanie wyniku jako prawdziwie pozytywne(PP) lub prawdziwie negatywne (PN)
                conf_matrix[index][index] +=1
            else:
                index = int(predictions[0][i]) #zakwalifikowanie wyniku jako fałszywie pozytywne (FP) lub fałszywie negatywne (FN)
                index2 = int(Y_test[i])
                conf_matrix[index][index2] +=1
        return conf_matrix
