import panel as pn

from ui.LoadEventLogPage import loadEventLogPage
from ui.FeatureSelectorPage import featureSelectorPage
from ui.SOMPage import SOMPage

Menu_List = []
Menu_List.append(('Event Log', loadEventLogPage))
Menu_List.append(('Feature Selection',featureSelectorPage))
Menu_List.append(('Self-Organinizing Map',SOMPage))
menu = pn.widgets.RadioButtonGroup(
    name='Menu',
    options=[item[0] for item in Menu_List],
    orientation='vertical',
    sizing_mode='stretch_width',
    styles={'font-size': '18pt', 'height': '200pt'},
    margin=(5, 0, 5, 0)
)
main_area = pn.Column(*[item[1] for item in Menu_List])

# Set all but the first page invisible
for i in range(len(Menu_List)-1):
    Menu_List[i+1][1].visible = False

def menu_picker(choice):
    for i, menu_item in enumerate(Menu_List):
        menu_item[1].visible = False # first make all invisible (otherwise two pages could be visible simultaneously for a split-second)
        if menu_item[0] == choice:
            make_visible = menu_item[1]
    make_visible.visible = True
pn.bind(menu_picker, menu, watch=True)

# will be served from main.py
template = pn.template.MaterialTemplate(
    site="SOM in Process Mining",
    title="by Max Tröger",
    sidebar=[menu],
    main=[main_area],
    sidebar_width = 280
)

template.servable()