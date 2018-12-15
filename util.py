
def split_data(tr_index, test_index, x , y, x_test, y_test):
    x_train = x[tr_index,: ].reshape(1,-1)
    y_train = y[tr_index].reshape(1,)
    
    x_test = x_test[test_index,: ].reshape(1,-1)
    y_test = y_test[test_index].reshape(1,)
    
    return x_train, y_train, x_test, y_test

def split_data_k(tr_indices, test_index, x , y, x_test, y_test):
    x_train = x[tr_indices,: ].reshape(len(tr_indices),-1)
    y_train = y[tr_indices].reshape(len(tr_indices),)
    
    x_test = x_test[test_index,: ].reshape(1,-1)
    y_test = y_test[test_index].reshape(1,)
    
    return x_train, y_train, x_test, y_test