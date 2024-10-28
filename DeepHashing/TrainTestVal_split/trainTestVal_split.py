# %%
from tensorflow import keras

# %%
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# %%
valSize = 10000

# %%
X_val = X_train[-valSize:]
y_val = y_train[-valSize:]

X_train = X_train[:-valSize]
y_train = y_train[:-valSize]

# %%
if __name__ == "__main__":
    print("X_train:", len(X_train))
    print("X_val:"  , len(X_val)  )
    print("X_test:" , len(X_test) )



