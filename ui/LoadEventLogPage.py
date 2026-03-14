import panel as pn
import param

from core.Logger import logger
import core.EventLog as EventLog

class LoadingState(param.Parameterized):
    state = param.Selector(objects=["NothingDone", "Loading", "EventLogLoaded", "DeclareCalculated", "NoFiles", "MultipleFiles", "Error"])
loadingState = LoadingState()

def delete(event):
    loadingState.state = "NothingDone"
    EventLog.delete()

def p_Status_determineStatusText(state):
    if state == "NothingDone":
        return "Please select an event log and press the button."
    elif state == "Loading":
        return "Loading..."
    elif state == "EventLogLoaded":
        return "Event log loaded."
    elif state == "DeclareCalculated":
        return "Declare Constraints calculated."
    elif state == "NoFiles":
        return "No file is selected! Please select an event log and try again."
    elif state == "MultipleFiles":
        return "Multiple files are selected! Please select only one and try again."
    elif state == "Error":
        return "An Error occured during loading!"

def p_Status_determineStatusStyle(state):
    if state in ("NothingDone", "Loading"):
        color = 'olive'
    elif state in ("EventLogLoaded", "DeclareCalculated"):
        color = 'green'
    else:
        color = 'red'
    return {'font-size': '16pt', 'color': color}

def w_FileLoadButton_action(event):
    with w_FileLoadButton.param.update(disabled=True):
        if len(w_FileSelector.value) == 0:
            loadingState.state = "NoFiles"
        elif len(w_FileSelector.value) > 1:
            loadingState.state = "MultipleFiles"
        else:
            loadingState.state = "Loading"
            EventLog.load_from_file(w_FileSelector.value[0])
            loadingState.state = "EventLogLoaded"

def w_AttributeSelect_determineOptions(state):
    if state == "EventLogLoaded":
        return EventLog.log.event_attributes
    return ["(nothing here yet)"]

def w_AggregationSelect_determineOptions(state, attribute_selected):
    if state == "EventLogLoaded":
        return EventLog.log.get_aggregation_operations(attribute_selected)
    return ["(nothing here yet)"]

def l_EventAttributeCard_visible(state):
    if state == "EventLogLoaded":
        return True
    else:
        return False

def w_AggregationButton_action(event):
    with w_AggregationButton.param.update(disabled=True):
        EventLog.log.add_event_attribute(False, aggregation=w_AggregationSelect.value, attribute=w_AttributeSelect.value)

def w_CustomAggregationButton_action(event):
    with w_CustomAggregationButton.param.update(disabled=True):
        EventLog.log.add_event_attribute(True, customAggName=w_CustomAggregationText.value, customAggCode=w_CustomAggregationTextArea.value)


# 1. p_Status
p_Status = pn.pane.Str(pn.bind(p_Status_determineStatusText, loadingState.param['state']),
                       styles=pn.bind(p_Status_determineStatusStyle, loadingState.param['state']))

# 2. l_FileSelectorCard
w_FileSelector = pn.widgets.FileSelector(file_pattern="*.xes", directory="inputs", only_files=True)
w_FileLoadButton = pn.widgets.Button(name="Load Event Log", button_type="primary")
w_FileLoadButton.on_click(w_FileLoadButton_action)
l_FileSelectorCard = pn.Card(w_FileSelector, w_FileLoadButton, title="Load Event Log", collapsed=False)

# 3. l_EventAttributeCard
w_AttributeSelect = pn.widgets.Select(name='Event Attribute', options=pn.bind(w_AttributeSelect_determineOptions, loadingState.param['state']))
w_AggregationSelect = pn.widgets.Select(name='Aggregation Operation', options=pn.bind(w_AggregationSelect_determineOptions, loadingState.param['state'], w_AttributeSelect))
w_AggregationButton = pn.widgets.Button(name='Aggregate', button_type='primary')
w_AggregationButton.on_click(w_AggregationButton_action)
w_CustomAggregationText = pn.widgets.TextInput(name='Name your aggregation', value='MyOwnAggregation')
w_CustomAggregationTextArea = pn.widgets.TextAreaInput(name='def CustomAggregationFunction(trace):', width=500, height=250, placeholder=f'#Example code\nif "A_DECLINED" in trace["concept:name"].values:\n    return "DECLINED"\nelif "A_APPROVED" in trace["concept:name"].values:\n    return "APPROVED"\nelif "A_CANCELLED" in trace["concept:name"].values:\n    return "CANCELLED"\nelse:\n    return "UNDECIDED"')
w_CustomAggregationButton = pn.widgets.Button(name='Aggregate', button_type='primary')
w_CustomAggregationButton.on_click(w_CustomAggregationButton_action)
l_EventAttributeCard = pn.Card(
    pn.pane.Markdown("Event attributes must be aggregated on a case-based level to be used by the tool. All aggregations you did here are listed under \"Event Attributes\" in the Feature Selection tab.\n\n**Normal Aggregations:** Select an attribute and an aggregation, and click 'Aggregate'."),
    pn.Row(w_AttributeSelect, w_AggregationSelect, w_AggregationButton),
    pn.layout.Divider(),
    pn.pane.Markdown("**Custom Aggregation:** Insert a name for your custom aggregation and insert Python code to define your aggregation function. The function takes a pandas DataFrame 'trace' as input and must return a string or a numeric value. An example is given in the text area."),
    pn.Row(w_CustomAggregationText, w_CustomAggregationButton),
    w_CustomAggregationTextArea,
    title="Aggregation of Event Attributes",
    collapsed=False,
    visible=pn.bind(l_EventAttributeCard_visible, loadingState.param['state'])
)

# 4. w_DeleteButton
w_DeleteButton = pn.widgets.Button(name='Delete', button_type='primary')
w_DeleteButton.on_click(delete)

loadEventLogPage = pn.Column(
    "### Load",
    p_Status,
    l_FileSelectorCard,
    l_EventAttributeCard,
    w_DeleteButton
)