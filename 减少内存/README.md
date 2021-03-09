# Code
```python
def reduce_memory(data):
    start_memory = data.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_memory," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    
    for col in data.columns:
        if ('int' in data[col].dtype.name) or ('float' in data[col].dtype.name):  # Exclude strings
            try:
                # Print current column type
                print("******************************")
                print("Column: ",col)
                print("dtype before: ",data[col].dtype)

                # make variables for Int, max and min
                IsInt = False
                value_max = data[col].max()
                value_min = data[col].min()

                # Integer does not support NA, therefore, NA needs to be filled
                if not np.isfinite(data[col]).all(): 
                    NAlist.append(col)
                    data[col].fillna(value_min-1,inplace=True)  

                # test if column can be converted to an integer
                asint = data[col].fillna(0).astype(np.int64)
                result = (data[col] - asint)
                result = result.sum()
                if result > -0.01 and result < 0.01:
                    IsInt = True


                # Make Integer/unsigned Integer datatypes
                if IsInt:
                    if value_min >= 0:
                        if value_max < 255:
                            data[col] = data[col].astype(np.uint8)
                        elif value_max < 65535:
                            data[col] = data[col].astype(np.uint16)
                        elif value_max < 4294967295:
                            data[col] = data[col].astype(np.uint32)
                        else:
                            data[col] = data[col].astype(np.uint64)
                    else:
                        if value_min > np.iinfo(np.int8).min and value_max < np.iinfo(np.int8).max:
                            data[col] = data[col].astype(np.int8)
                        elif value_min > np.iinfo(np.int16).min and value_max < np.iinfo(np.int16).max:
                            data[col] = data[col].astype(np.int16)
                        elif value_min > np.iinfo(np.int32).min and value_max < np.iinfo(np.int32).max:
                            data[col] = data[col].astype(np.int32)
                        elif value_min > np.iinfo(np.int64).min and value_max < np.iinfo(np.int64).max:
                            data[col] = data[col].astype(np.int64)    

                # Make float datatypes 32 bit
                else:
                    data[col] = data[col].astype(np.float32)

                # Print new column type
                print("dtype after: ",data[col].dtype)
                print("******************************")
            except:
                print("dtype after: Failed")
        else:
            print("dtype remain: ",data[col].dtype)
    
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    end_memory = data.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",end_memory," MB")
    print("This is ",100*start_memory/end_memory,"% of the initial size")
    print("Missing Value list", NAlist)
    return data
```
