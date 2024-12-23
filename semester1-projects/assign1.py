#these are psudocodes for different algorithms
def linear_simple(processed_data):
  X, Y, U, X_test = processed_data
  XTX = X.T @ (X * U)
  XTY = X.T @ (Y * U)
  w_star = np.linalg.solve(XTX, XTY)
  Y_pred = X_test @ w_star

def linear_regularised():
  XTX_regularised = X.T @ X + best_lambda * I
  XTY = X.T @ Y
  w_final = linalg.solve(XTX_regularised, XTY)
  Y_pred = X_test @ w_final

def logistic_binary(X, Y):
  W = np.random.randn(n_features)
  sig = 1/ (1 + np.exp(-X@W))
  loss  = -np.sum(Y * np.log(sig) + (1 - Y) * np.log(1 - sig))
  W_final = gradient_decent(X, Y, W, learning_rate)
  Y_pred = 1/ (1 + np.exp(-X@W_final)) >= 0.5

def logistic_multi(X, Y):
  W = np.random.randn(n_features, n_classes)
  softmax = np.exp(X@W)/np.sum(np.exp(X@W))
  loss  = sum( log(softmax where y==1)
  W_final = gradient_decent(X, Y, W, learning_rate)
  Y_pred = np.exp(X@W_final)/np.sum(np.exp(X@W_final))

def neural_net():
  def forward():
    Zn = An-1@Wn+bn
    An = sigmoid(zn)
  def backward():
    dan = y/y_hat - (1-y)/(1-y_hat)
    dzn = dan.an*(1-an) = y - y_hat 
    dwn = dzn.an-1
    db1 = dzn.1
    
    dan-1 = dzn@Wn.T

    W -= rate*dw
    b -= rate*db
    dw = dan.sig(an)
      

