import panel as pn
import param
import numpy as np
import pandas as pd
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# Moved some variables and functions to this file so that the processes of ProcessPoolExecutor don't have to import everything imported here
from core.DeclareConstraintsAlgorithm import ONE_VAR_CONSTRAINTS, ONE_VAR_LEN, TWO_VAR_CONSTRAINTS_PARTNERS, TWO_VAR_CONSTRAINTS, TWO_VAR_LEN, _YPos, _YPosInv, _isOneVar, _Declare

class DeclareConstraint:
    # some constants
    A = 0 # number of unique activities
    Asq = 0
    N_DeclareFeatures = 0
    
    # reference to Log object
    log = None
    
    df_declare = None # will be copied into log

    def __init__(self, log):
        self.A = log.N_Activities
        self.Asq = self.A ** 2
        self.N_DeclareFeatures = ONE_VAR_LEN*self.A + TWO_VAR_LEN*self.Asq
        self.log = log

    def calculateDeclareFeatures(self, _callback_function=None):
        def callback_function(x):
            if _callback_function is not None:
                _callback_function(x)
        
        data = np.empty((self.log.N_Cases, self.N_DeclareFeatures), dtype=np.int8)
        data_index = np.empty(self.log.N_Cases, dtype=int)

        with ProcessPoolExecutor() as executor:
            futures = []
            for case_i, trace_df in self.log.df_log.groupby(self.log.CASE):
                trace = trace_df[self.log.ACTIVITY].to_list()
                futures.append(executor.submit(_Declare, self.A, self.Asq, self.N_DeclareFeatures, case_i, trace))
            counter = 0
            callback_time = time.time()
            for future in as_completed(futures):
                case_i, y = future.result()
                data_index[case_i] = case_i # case_i are integers 0, 1, 2, ...
                data[case_i,:] = y
                # Call the callback function every 200 ms
                counter += 1
                if time.time() >= callback_time + 0.2:
                    callback_time = time.time()
                    callback_function(counter/self.log.N_Cases * 0.5)
        self.df_declare = pd.DataFrame(data, index=data_index, columns=list(range(self.N_DeclareFeatures)))

        # Delete all constraints with a==b:
        delete_indices = []
        for constraint in TWO_VAR_CONSTRAINTS:
            delete_indices += self.get_duplicate_indices(constraint)
        self.df_declare.drop(delete_indices, axis=1, inplace=True)
    
    def get_train_columns(self, constraint):
        if constraint in ONE_VAR_CONSTRAINTS:
            start = self.YPos(constraint, 0)
            end = self.YPos(constraint, self.A-1)
            train_columns = [*range(start, end+1)]
        else:
            start = self.YPos(constraint, 0, 0)
            end = self.YPos(constraint, self.A-1, self.A-1)
            train_columns = [*range(start, end+1)]
            for dupl_index in self.get_duplicate_indices(constraint):
                train_columns.remove(dupl_index)
        return train_columns
    
    def get_duplicate_indices(self, constraint):
        duplicate_indices = []
        if constraint not in ONE_VAR_CONSTRAINTS:
            for i in range(self.A):
                duplicate_indices.append(self.YPos(constraint, i, i))
        return duplicate_indices

    def getValue(self, case, constraint):
        constraint_name = constraint[0]
        a = constraint[1]
        if len(constraint) == 2:
            constraint_number = self.YPos(constraint_name, a)
        elif len(constraint) == 3:
            b = constraint[2]
            if a == b:
                return np.nan
            constraint_number = self.YPos(constraint_name, a, b)
        else:
            raise ValueError(f"Length of 'constraint' must be 2 or 3, but is {len(constraint)}.")
        return self.df_declare.loc[case, constraint_number]

    def getFeatureName(self, number):
        constraint, a, b = self.YPosInv(number)
        if b is None:
            return f"{constraint}({self.log.activity_mapping_inv[a]})"
        else:
            return f"{constraint}({self.log.activity_mapping_inv[a]}, {self.log.activity_mapping_inv[b]})"

    def getPartner(self, number):
        constraint, a, b = self.YPosInv(number)
        
        for c, isFirst in TWO_VAR_CONSTRAINTS_PARTNERS:
            if c == constraint:
                if isFirst:
                    aPartner = a
                else:
                    aPartner = b
                break
        else: # else-part will only be executed if for loop is not breaked
            raise ValueError(f"The constraint {constraint} is no two-variable constraint!")
        
        constraintPartner = "AtLeastOne"
        return self.YPos(constraintPartner, aPartner)

    def YPos(self, constraint, a, b=None):
        return _YPos(self.A, self.Asq, constraint, a, b=b)

    def YPosInv(self, number):
        return _YPosInv(self.A, self.Asq, number)

    def isOneVar(self, number):
        return _isOneVar(self.A, self.Asq, number)

