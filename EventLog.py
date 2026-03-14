import panel as pn
import param
import pm4py
import numpy as np
import pandas as pd
import time

from core.Logger import logger
from core.DeclareConstraints import DeclareConstraint
from ui.FeatureSelectorPage import featureSelector_build, featureSelector_addEventAttribute, featureSelector_RESET

log = None

def load_from_file(file_path):
    global log
    delete()
    log = Log(file_path)

def delete():
    global log
    if log is not None:
        log = None
    featureSelector_RESET()

class Log:
    ACTIVITY = "concept:name"
    CASE = "case:concept:name"
    TIMESTAMP = "time:timestamp"

    df_data = None
    # df_data column meanings:
    # "DECL_" = declare constraints
    # "MAIN_" = main features (duration, NEvents)
    # "EATT_" = event attributes
    # "CATT_" = case attributes
    df_columns = None # information about all columns in df_data except "DECL_"; columns: is_numeric, support

    df_log = None
    df_log_numeric = None # the same log, but everything is numeric (or nan if conversion wasn't possible)
    groupby_df_log = None
    groupby_df_log_numeric = None
    
    event_attributes = None # list of column names
    
    activity_mapping = None
    activity_mapping_inv = None
    caseId_mapping = None
    caseId_mapping_inv = None

    N_Activities = 0
    N_Cases = 0

    activities_num = None
    activities = None
    cases_num = None
    cases = None
    
    declareConstraint = None # DeclareConstraint object

    def __init__(self, file_path):
        self.df_log = pm4py.read_xes(file_path, variant="rustxes")
        
        # replace activity names and case names with integers
        self.activity_mapping = {name: i for i, name in enumerate(sorted(self.df_log[self.ACTIVITY].unique()))}
        self.activity_mapping_inv = {i: name for name, i in self.activity_mapping.items()}
        self.caseId_mapping = {name: i for i, name in enumerate(sorted(self.df_log[self.CASE].unique()))}
        self.caseId_mapping_inv = {i: name for name, i in self.caseId_mapping.items()}
        
        self.N_Activities = len(self.activity_mapping)
        self.N_Cases = len(self.caseId_mapping)
        
        self.activities_num = list(range(self.N_Activities))
        self.activities = [self.activity_mapping_inv[i] for i in self.activities_num]
        self.cases_num = list(range(self.N_Cases))
        self.cases = [self.caseId_mapping_inv[i] for i in self.cases_num]
        
        self.df_log[self.ACTIVITY] = self.df_log[self.ACTIVITY].map(self.activity_mapping)
        self.df_log[self.CASE] = self.df_log[self.CASE].map(self.caseId_mapping)
        self.df_log_numeric = self.df_log.apply(pd.to_numeric, errors="coerce")
        
        # Create df_data, df_columns with Duration and NEvent data
        grouped = self.df_log[[self.CASE, self.TIMESTAMP]].groupby(self.CASE, sort=True).agg(["size", "min", "max"])
        self.df_data = pd.DataFrame({"MAIN_Duration":(grouped[self.TIMESTAMP,"max"] - grouped[self.TIMESTAMP,"min"]).dt.total_seconds(), "MAIN_NEvents":grouped[self.TIMESTAMP,"size"]})
        self.df_columns = pd.DataFrame({"is_numeric":[True,True], "support":[1.0,1.0]}, index=["MAIN_Duration", "MAIN_NEvents"])
        
        self.groupby_df_log = self.df_log.groupby(self.CASE, sort=True)
        self.groupby_df_log_numeric = self.df_log_numeric.groupby(self.CASE, sort=True)
        is_case_attribute = (self.groupby_df_log.nunique() <= 1).all() # NaNs are not counted in .nunique()
        # Example of is_case_attribute:
        # case:REG_DATE            True
        # concept:name            False
        # org:resource            False
        # case:AMOUNT_REQ          True
        # time:timestamp          False
        # lifecycle:transition    False
        # dtype: bool
        
        self.event_attributes = []
        for column in self.df_log.columns:
            if column in (self.CASE, self.TIMESTAMP, self.ACTIVITY):
                continue
            if is_case_attribute[column]:
                c = "CATT_"+column
                if not self.df_log_numeric[column].isna().all(): # numeric
                    self.df_data[c] = self.groupby_df_log_numeric[column].first()
                    support = self.df_data[c].notna().mean()
                    self.df_columns.loc[c] = {"is_numeric":True,"support":support} # .loc[c] adds a new row with index c
                else: # not numeric
                    self.df_data[c] = self.groupby_df_log[column].first()
                    support = self.df_data[c].notna().mean()
                    self.df_columns.loc[c] = {"is_numeric":False,"support":support} # .loc[c] adds a new row with index c
            else:
                self.event_attributes.append(column)
        
        # initialize DeclareConstraint object and calculate Declare constraints
        self.declareConstraint = DeclareConstraint(self)
        self.declareConstraint.calculateDeclareFeatures()
        self.df_data = pd.concat([self.df_data, self.declareConstraint.df_declare.add_prefix("DECL_")], axis=1)
        
        featureSelector_build(self)
    
    # called by LoadEventLogPage
    def get_aggregation_operations(self, event_attribute):
        if event_attribute is None:
            return []
        if not self.df_log_numeric[event_attribute].isna().all(): # numeric
            return ["count", "nunique", "first", "last", "min", "max", "mean", "median", "sum", "prod", "std", "var", "skew"]
        else: # not numeric
            return ["count", "nunique", "first", "last"]
    
    # called by LoadEventLogPage
    def add_event_attribute(self, customAgg, aggregation=None, attribute=None, customAggName=None, customAggCode=None):
        if customAgg:
            c = "EATT_CustomAgg_" + customAggName
            customAggFunction = createCustomAggregationFunction(customAggCode)
            is_numeric = False
            
            self.df_log[self.ACTIVITY] = self.df_log[self.ACTIVITY].map(self.activity_mapping_inv) # Change activities back to their real names
            groupby_df_log = self.df_log.groupby(self.CASE, sort=True) # we have to create a new groupby object, because self.groupby_df_log uses the old values of self.df_log[self.ACTIVITY]
            self.df_data[c] = groupby_df_log.apply(customAggFunction, include_groups=False)
            self.df_log[self.ACTIVITY] = self.df_log[self.ACTIVITY].map(self.activity_mapping)
        else:
            c = "EATT_" + aggregation + "_" + attribute
            if c in log.df_columns.index.to_list(): # don't add twice
                return
            if not self.df_log_numeric[attribute].isna().all(): # numeric
                is_numeric = True
                self.df_data[c] = getattr(self.groupby_df_log_numeric[attribute], aggregation)()
            else: # not numeric
                is_numeric = False
                self.df_data[c] = getattr(self.groupby_df_log[attribute], aggregation)()
        
        support = self.df_data[c].notna().mean()
        self.df_columns.loc[c] = {"is_numeric":is_numeric,"support":support}
        featureSelector_addEventAttribute(log, c)

def createCustomAggregationFunction(code):
    namespace = {}
    
    # indent by 4 spaces
    indented_code = "\n".join([f"    {line}" for line in code.splitlines()])
    full_code = f"def CustomAggregationFunction(trace):\n{indented_code}"
    # DANGER! THIS IS VERY UNSECURE! IT LETS THE USER RUN ARBITRARY PYTHON CODE ON THIS MACHINE!
    exec(full_code, namespace)
    
    return namespace['CustomAggregationFunction']