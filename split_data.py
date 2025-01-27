import index
import sklearn

print("split_data.py script is running...")

df_lencoder = index.df_lencoder

# Memisahkan fitur (X) dan target (y)
X = df_lencoder.drop(columns=['SalePrice'])  
y = df_lencoder['SalePrice'] 

from sklearn.model_selection import train_test_split
 
# membagi dataset menjadi training dan testing 
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# menghitung panjang/jumlah data 
print("Jumlah data: ",len(X))
# menghitung panjang/jumlah data pada x_test
print("Jumlah data latih: ",len(x_train))
# menghitung panjang/jumlah data pada x_test
print("Jumlah data test: ",len(x_test))
