import panel as pn
from datetime import datetime

from core.DeclareConstraintsAlgorithm import ONE_VAR_CONSTRAINTS, TWO_VAR_CONSTRAINTS
import core.EventLog as EventLog

RADIO_OPTIONS = {"Yes":True, "No":False}

list_MAIN = []
list_CATT = []
list_EATT = []
list_DECL_ONE = []
list_DECL_TWO = []

def featureSelector_RESET():
    global list_MAIN, list_CATT, list_EATT, list_DECL_ONE, list_DECL_TWO
    list_MAIN = []
    list_CATT = []
    list_EATT = []
    list_DECL_ONE = []
    list_DECL_TWO = []
    l_MAIN_Card[0].clear()
    l_CATT_Card[0].clear()
    l_EATT_Card[0].clear()
    l_DECL_Card[0].clear()

class SelectionElement:
    name = None
    radio_button_group = None
    row = None
    
    def __init__(self, name, is_selectable, support=None):
        self.name = name
        
        # Part 1
        PART1_WIDTH = 100
        if is_selectable:
            part1 = pn.widgets.RadioButtonGroup(options=RADIO_OPTIONS, value=False, width=PART1_WIDTH)
            self.radio_button_group = part1
        else:
            part1 = pn.pane.Str("Not selectable", width=PART1_WIDTH)
            
        # Part 2
        PART2_WIDTH = 150
        if support is not None:
            part2 = pn.pane.Str(f"Support: {support*100:.3f}%", width=PART2_WIDTH)
        else:
            part2 = pn.pane.Str("", width=PART2_WIDTH)
            
        # Part 3
        part3 = pn.pane.Str(name)
        
        self.row = pn.Row(part1, part2, part3)
    
    def is_train_selected(self):
        if self.radio_button_group is None:
            return False
        else:
            return self.radio_button_group.value

# is called by EventLog
def featureSelector_build(log):
    # Main Attributes
    columns = [c for c in log.df_columns.index.to_list() if c.startswith("MAIN_")]
    for column_name in columns:
        selectionElement = SelectionElement(column_name, log.df_columns.loc[column_name, "is_numeric"], log.df_columns.loc[column_name, "support"])
        list_MAIN.append(selectionElement)
        l_MAIN_Card[0].append(selectionElement.row)
    
    # Case Attributes
    columns = [c for c in log.df_columns.index.to_list() if c.startswith("CATT_")]
    for column_name in columns:
        selectionElement = SelectionElement(column_name, log.df_columns.loc[column_name, "is_numeric"], log.df_columns.loc[column_name, "support"])
        list_CATT.append(selectionElement)
        l_CATT_Card[0].append(selectionElement.row)
    
    # Declare Constraints
    declare_column = pn.Column()
    for constraint in ONE_VAR_CONSTRAINTS:
        selectionElement = SelectionElement(constraint, True)
        list_DECL_ONE.append(selectionElement)
        l_DECL_Card[0].append(selectionElement.row)
    
    for constraint in TWO_VAR_CONSTRAINTS:
        selectionElement = SelectionElement(constraint, True)
        list_DECL_TWO.append(selectionElement)
        l_DECL_Card[0].append(selectionElement.row)

# is called by EventLog
def featureSelector_addEventAttribute(log, column_name):
    selectionElement = SelectionElement(column_name, log.df_columns.loc[column_name, "is_numeric"], log.df_columns.loc[column_name, "support"])
    list_EATT.append(selectionElement)
    l_EATT_Card[0].append(selectionElement.row)

# is called by SOM
def get_train_data():
    all_train_columns = []
    # MAIN_, CATT_, and EATT_ columns
    for selectionElement in list_MAIN + list_CATT + list_EATT:
        if selectionElement.is_train_selected():
            all_train_columns.append(selectionElement.name)
    
    # DECL_ columns
    for selectionElement, constraint in zip(list_DECL_ONE, ONE_VAR_CONSTRAINTS):
        if selectionElement.is_train_selected():
            all_train_columns += ["DECL_" + str(i) for i in EventLog.log.declareConstraint.get_train_columns(constraint)]
    for selectionElement, constraint in zip(list_DECL_TWO, TWO_VAR_CONSTRAINTS):
        if selectionElement.is_train_selected():
            all_train_columns += ["DECL_" + str(i) for i in EventLog.log.declareConstraint.get_train_columns(constraint)]
    
    return EventLog.log.df_data[all_train_columns]

l_MAIN_Card = pn.Card(pn.Column(), title="Main Attributes", collapsed=False)
l_CATT_Card = pn.Card(pn.Column(), title="Case Attributes", collapsed=False)
l_EATT_Card = pn.Card(pn.Column(), title="Event Attributes", collapsed=False)
l_DECL_Card = pn.Card(pn.Column(), title="Declare Constraints", collapsed=False)

featureSelectorPage = pn.Column("## Feature Selection", pn.Column("Select your training data:", l_MAIN_Card, l_CATT_Card, l_EATT_Card, l_DECL_Card))