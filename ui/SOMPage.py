import numpy as np
import bokeh
import bokeh.plotting # It is need, but I do not know why
import panel as pn

import core.EventLog as EventLog
import core.SOM as SOM
from core.Logger import logger

import time

COLOR_SkyBlue = np.array([86, 180, 233])
COLOR_Orange = np.array([230, 159, 0])
COLOR_White = np.array([255, 255, 255])
COLOR_Error = "#b600ff"

# all these palettes must the numpy arrays AND have the color for outliers as their (additional) last color
PALETTE_Standard = np.array(list(bokeh.palettes.Greys256[::-1]) + [COLOR_Error])

PALETTE_DECL = [] # create myself a sky-blue/orange palette
for x in np.linspace(0,1,256):
    color = (x*COLOR_SkyBlue + (1-x)*COLOR_Orange).astype(int)
    PALETTE_DECL.append(bokeh.colors.RGB(*color).to_hex())
PALETTE_DECL = np.array(PALETTE_DECL + [COLOR_Error])

PALETTE_Feature = np.array(list(bokeh.palettes.Viridis256) + [COLOR_Error])

# other types of palettes
PALETTE_CaseFeatures_Tens = bokeh.palettes.Viridis[10]
PALETTE_Categorical = bokeh.palettes.Bright

def Get_Colors(palette, np_values): # np_values should lie between 0.0 and 1.0 (other values will get the color for outliers)
    length = len(palette) - 1
    color_indices = np.clip( (np_values*(length-0.01)).astype(int) , -1 , length ) # Subtract 0.01 for making a value of exactly 1.0 not the color for outliers
    return palette[color_indices]                                                  # Both indices -1 and 256 lead to the outlier color

HOVER_TOOLTIPS_Standard = [("support", "@support")]

COLORBAR_OFFSET_FACTOR = 1.0001 # this factor extends the colorbars maximal value to avoid values just a bit too high

SOM_plots = []
previous_SOM_width = -1

class SOM_plot:
    index = -1
    
    OnOff_select = None
    feature_select = None
    cluster_select = None
    
    def __init__(self):
        OnOff_options = ["Support Circles"]
        self.OnOff_select = pn.widgets.CheckButtonGroup(value=[], options=OnOff_options, orientation='horizontal')
        
        # Feature Select
        self.feature_select = pn.widgets.Select(name="Feature to visualize", width=1000)
        self.feature_select.options = select_options_names
        self.feature_select.value = -1
        # Cluster Select
        self.cluster_select = pn.widgets.IntInput(name='# of Clusters', width=120, value=3, start=3, end=10,
                                                  visible=pn.bind(lambda feature: True if feature in (-5,-6) else False, self.feature_select))
        
        # Row
        row = pn.Row(self.OnOff_select, self.feature_select, self.cluster_select)
        SOMPage[2].append(row)
        pn.bind(lambda _1,_2,_3: update_feature_visualization([self]), self.OnOff_select, self.feature_select, self.cluster_select, watch=True) # this bind has to be done here since otherwise feature_select doesn't work :(

def create_new(event):
    # +++ Data +++
    global hex_X, hex_Y, q, r, n, one_dim_mode, source, source_pie
    hex_X = w_X.value
    hex_Y = w_Y.value
    SOM.create_new(hex_X, hex_Y, w_Init.value)
    q = SOM.arr_MCoords[:,0] + np.sqrt(3)/3 * SOM.arr_MCoords[:,1] # coordinate transformation
    r = -2*np.sqrt(3)/3*SOM.arr_MCoords[:,1]
    n = len(q)
    one_dim_mode = True if hex_Y == 1 else False
    
    # +++ Sources +++
    source = bokeh.plotting.ColumnDataSource({'x':[],'y':[],'q':[],'r':[],'circles_r':[],'circles_alpha':[],'color':[],'value':[],'value2':[],'support':[]})
    source_pie = bokeh.plotting.ColumnDataSource({key:[] for key in SOURCE_PIE_KEYS})
    
    # +++ Figure +++
    global p
    p = bokeh.plotting.figure(sizing_mode='stretch_width', height=600, match_aspect=True, tools="pan,wheel_zoom,reset", background_fill_color="#000000", output_backend="webgl")
    p.xgrid.visible = False
    p.ygrid.visible = False
    p.axis.visible = False
    if one_dim_mode:
        p.rect(x='x', y='y', fill_color='color', line_color='color', source=source, width=1, height=1, alpha=1)
    else:
        p.hex_tile('q', 'r', fill_color='color', line_color='color', source=source, size=np.sqrt(3)/3, alpha=1)
    p.annular_wedge(x='x', y='y', outer_radius=0.5, inner_radius='radius', start_angle='start_angle', end_angle='end_angle',
                                     color='color', fill_alpha=1, line_alpha=0, source=source_pie)
    p.circle('x', 'y', radius='circles_r', alpha='circles_alpha', source=source, fill_color='#000000', line_color=None)
    # TODO Hover:
    #hover = bokeh.models.HoverTool(renderers=[self.hextile])
    #p.add_tools(self.hover)
    SOMPage[3] = p
    SOMPage[2] = pn.Column()
    
    
    # Create select_options dictionary
    # select_options:
    # >=0: train data
    # -1: U-Matrix
    # -2: nothing
    # -5: K-Means Clustering
    # -6: Agglomerative Clustering
    # <=-7: test data
    global select_options, select_options_inv, test_data_id_start, select_options_names
    select_options = {"Agglomerative Clustering":-6, "KMeans Clustering":-5, "(nothing)":-2, "U-Matrix":-1}
    test_data_id_start = -7
    _all_columns = EventLog.log.df_data.columns.to_list()
    _train_columns = SOM.df_train.columns.to_list()
    _test_columns = [c for c in _all_columns if c not in _train_columns]
    i = 0
    for c in _train_columns:
        select_options[c] = i
        i += 1
    i = test_data_id_start
    for c in _test_columns:
        select_options[c] = i
        i -= 1
    select_options_inv = {b:a for a,b in select_options.items()}
    select_options_names = {}
    for key, value in select_options.items():
        if key.startswith("DECL_"):
            DECL_number = int(key.split("_")[1])
            key = "DECL_" + str(EventLog.log.declareConstraint.getFeatureName(DECL_number))
        if value >= 0:
            key = f"[TRAIN] {key}"
        elif value <= test_data_id_start:
            key = f"[TEST] {key}"
        select_options_names[key] = value
    
    # +++ SOM_plot +++
    global SOM_plots
    SOM_plots = []
    add_new_plot(amount_override=1)

def add_new_plot(event=None, amount_override=None):
    amount = w_NewSOMPlotAmount.value
    if amount_override is not None:
        amount = amount_override
    
    # add new lines to source
    new_source = source.data.copy() # dictionary
    new_source['x'].extend([0.0]*amount*n)
    new_source['y'].extend([0.0]*amount*n)
    new_source['q'].extend([0.0]*amount*n)
    new_source['r'].extend([0.0]*amount*n)
    new_source['circles_r'].extend([0.0]*amount*n)
    new_source['circles_alpha'].extend([0.0]*amount*n)
    new_source['color'].extend([0.0]*amount*n)
    new_source['value'].extend([0.0]*amount*n) # for hover only I think
    new_source['value2'].extend([0.0]*amount*n) # for hover only I think
    new_source['support'].extend([0.0]*amount*n) # for hover only I think
    source.data = new_source # all columns must be replaced at once if their length changes
    
    new_SOM_plots = []
    for i in range(amount):
        som_plot = SOM_plot() # takes about 1s and even longer for every other existing som_plot
        SOM_plots.append(som_plot)
        new_SOM_plots.append(som_plot)
    update_feature_visualization(new_SOM_plots) # takes about 0.1s per plot and even longer for every other existing som_plot

def update_multi(event):
    update(event, multi=True)

def update(event, multi=False):
    with w_TrainButton.param.update(disabled=True):
        with w_TrainButtonMulti.param.update(disabled=True):
            if multi:
                SOM.train_multi(w_Sigma_start.value, w_Sigma_end.value, w_Epochs.value)
            else:
                SOM.train(w_Sigma.value)
            update_feature_visualization(SOM_plots)
            
            # Error text
            if SOM.quantization_error_2 is None:
                train_error_text.object = f"Quantization Error: {SOM.quantization_error_1:.5f}\nTopographic Error: {SOM.topographic_error*100:.5f} %"
            else:
                train_error_text.object = f"Quantization Error: {SOM.quantization_error_2:.5f} (euclidean),\n" \
                                          f"                    {SOM.quantization_error_1:.5f} (manhattan)\nTopographic Error: {SOM.topographic_error*100:.5f} %"

SOURCE_PIE_KEYS = ['x','y','radius','start_angle','end_angle','color']
class DonutMatrix:
    data = {} # dictionary of dictionaries
    
    def update(self, som_plot_index, data_dict):
        self.data[som_plot_index] = data_dict
    
    def remove(self, som_plot_index):
        if som_plot_index in self.data:
            del self.data[som_plot_index]
    
    def create_source(self):
        if len(self.data)==0:
            return {}
        new_source_pie = {key:[] for key in SOURCE_PIE_KEYS}
        for data_dict in self.data.values():
            for key, values in data_dict.items():
                new_source_pie[key].extend(values)
        return new_source_pie

donutMatrix = DonutMatrix()

def update_feature_visualization(som_plot_list, recalculate_positions=False):
    global previous_SOM_width
    if previous_SOM_width != w_MaxSOMWidth.value:
        previous_SOM_width = w_MaxSOMWidth.value
        update_feature_visualization(SOM_plots, recalculate_positions=True) # update everything
        return # don't run the function twice
    # +++ calculations used by all som_plots +++
    # hex_support and radius of black circles
    hex_support = SOM.get_hex_support()
    hex_support_min = np.min(hex_support[hex_support>0])
    hex_support_max = np.max(hex_support)
    MIN_AREA = 0.0025
    MAX_AREA = 0.09
    black_circle_area = np.zeros(n, dtype=float)
    if hex_support_max > hex_support_min:
        black_circle_area[hex_support>0] = (hex_support[hex_support>0] - hex_support_min) / (hex_support_max-hex_support_min) * (MAX_AREA-MIN_AREA) + MIN_AREA # min-max-scaling
    else:
        black_circle_area[hex_support>0] = MAX_AREA
    UMatrix = SOM.calculate_UMatrix()
    UMatrix_Max = np.max(UMatrix)
    
    new_source = source.data.copy()
    for som_plot in som_plot_list:
        # update x- and y-coordinates
        som_plot_index = SOM_plots.index(som_plot)
        I0 = som_plot_index*n
        I1 = (som_plot_index+1)*n
        x_trans = (som_plot_index % w_MaxSOMWidth.value) * (hex_X+2)
        if one_dim_mode: # one-dimensional SOM: Use different y spacing between plots
            y_trans = -(som_plot_index // w_MaxSOMWidth.value) * hex_Y
        else:
            y_trans = -(som_plot_index // w_MaxSOMWidth.value) * (hex_Y*np.sqrt(0.75)+2)
        # check index and possibly update locations
        if som_plot.index != som_plot_index or recalculate_positions:
            som_plot.index = som_plot_index
            _x_ = SOM.arr_MCoords[:,0] + x_trans
            _y_ = SOM.arr_MCoords[:,1] + y_trans
            new_source['x'][I0:I1] = _x_
            new_source['y'][I0:I1] = _y_
            new_source['q'][I0:I1] = _x_ + np.sqrt(3)/3 * _y_ # coordinate transformation
            new_source['r'][I0:I1] = -2*np.sqrt(3)/3 * _y_
        
        # do this everytime
        new_source['support'][I0:I1] = hex_support
        new_source['circles_r'][I0:I1] = np.sqrt(black_circle_area)
        if "Support Circles" in som_plot.OnOff_select.value:
            new_source['circles_alpha'][I0:I1] = [1.0]*n
        else:
            new_source['circles_alpha'][I0:I1] = [0.0]*n
        
        # Idea: Every feature has an if-part and an else-part. Only one if-part and all other else-parts will be executed.
        feature = som_plot.feature_select.value
        
        # (nothing)
        if feature == -2:
            new_source['value'][I0:I1] = np.zeros(n)
            new_source['color'][I0:I1] = ['#ffffff']*n
            #som_plot.hover.tooltips = HOVER_TOOLTIPS_Standard
        else:
            pass
        
        # U-Matrix
        if feature == -1:
            new_source['value'][I0:I1] = UMatrix
            new_source['color'][I0:I1] = Get_Colors(PALETTE_Standard, UMatrix / UMatrix_Max)
            #som_plot.hover.tooltips = HOVER_TOOLTIPS_Standard + [('U-Matrix', '@value')]
        else:
            pass
        
        # TRAIN DATA
        if feature >= 0:
            column_name = select_options_inv[feature]
            values = SOM.get_feature_values(column_name)
            new_source['value'][I0:I1] = values
            if column_name.startswith("DECL_"):
                declare_id = int(column_name[5:])
                doItOneStyle = True
                if not EventLog.log.declareConstraint.isOneVar(declare_id):
                    declare_id_AtLeastOne = EventLog.log.declareConstraint.getPartner(declare_id) # get the corresponding AtLeastOne feature
                    values2 = SOM.get_feature_values(f"DECL_{declare_id_AtLeastOne}")
                    if values2 is not None: # corresponding AtLeastOne is not here
                        doItOneStyle = False
                    else:
                        logger.info("AtLeastOne not included - two-variable Declare constraints will be colored with only two colors.")
                
                if doItOneStyle:
                    #som_plot.hover.tooltips = HOVER_TOOLTIPS_Standard + [(EventLog.log.declareConstraint.getFeatureName(declare_id), "@value")]
                    new_source['color'][I0:I1] = Get_Colors(PALETTE_DECL, values)
                else:
                    new_source['value2'][I0:I1] = values2
                    # calculate colors
                    colors = []
                    for i in range(len(values)):
                        x = values[i]
                        y = values2[i]
                        color = (1-x) * COLOR_Orange + (1-y) * COLOR_White + (x+y-1) * COLOR_SkyBlue
                        color = color.astype(int)
                        colors.append(bokeh.colors.RGB(*color).to_hex())
                    new_source['color'][I0:I1] = colors
                    #som_plot.hover.tooltips = HOVER_TOOLTIPS_Standard + [(EventLog.log.declareConstraint.getFeatureName(declare_id), "@value"), (EventLog.log.declareConstraint.getFeatureName(declare_id_AtLeastOne), "@value2")]
            else:
                #som_plot.hover.tooltips = HOVER_TOOLTIPS_Standard + [(column_name, "@value")]
                new_source['color'][I0:I1] = Get_Colors(PALETTE_Feature, values)
        else:
            pass
        
        # TEST DATA
        if feature <= test_data_id_start:
            column_name = select_options_inv[feature]
            b_DECL = False
            if column_name.startswith("DECL_"):
                b_DECL = True
                declare_id = int(column_name[5:])
                pie_value, num_classes = SOM.calculate_test_data_DECL(declare_id)
                if num_classes == 2:
                    tick_names = ["False", "True"]
                    ticks = [0.25, 0.75]
                else:
                    tick_names = ["False", "True but not activated", "True and activated"]
                    ticks = [1/6, 3/6, 5/6]
            elif EventLog.log.df_columns.loc[column_name, "is_numeric"]: # feature is numeric
                num_classes = 10
                pie_value, percentiles = SOM.calculate_test_data_numeric(column_name, num_classes) # shape = n * num_classes
                palette = PALETTE_CaseFeatures_Tens
                tick_names = [f"{p:.2f}" for p in percentiles]
                ticks = np.linspace(0, 1, num_classes+1) #[.0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.]
            else: # feature is categorical
                pie_value, num_classes, names_classes = SOM.calculate_test_data_categoric(column_name)
                palette = PALETTE_Categorical[num_classes if num_classes > 2 else 3] # TODO maybe there are too many classes
                tick_names = names_classes
                ticks = np.linspace(1/num_classes/2, 1-1/num_classes/2, num_classes)
            
            pie_total = np.sum(pie_value, axis=1) # shape = n
            MAX_AREA = 0.2
            inner_radius = np.zeros(n, dtype=float)
            if hex_support_max > hex_support_min:
                inner_radius[hex_support>0] = (hex_support_max-hex_support[hex_support>0]) / (hex_support_max-hex_support_min) * MAX_AREA # min-max-scaling (but with a twist)
            else:
                inner_radius[hex_support>0] = 0
            inner_radius = np.sqrt(inner_radius)
            new_pie_charts_data = {key:[] for key in SOURCE_PIE_KEYS}
            for i in range(pie_value.shape[0]):
                angle = 0
                for j in range(pie_value.shape[1]):
                    if pie_value[i,j] > 0:
                        new_pie_charts_data['x'].append(SOM.arr_MCoords[i,0] + x_trans)
                        new_pie_charts_data['y'].append(SOM.arr_MCoords[i,1] + y_trans)
                        new_pie_charts_data['radius'].append(inner_radius[i])
                        new_pie_charts_data['start_angle'].append(angle)
                        angle += pie_value[i,j] / pie_total[i] * 2 * np.pi
                        new_pie_charts_data['end_angle'].append(angle)
                        if b_DECL and num_classes == 2:
                            if j == 0:
                                new_pie_charts_data['color'].append(COLOR_Orange)
                            elif j == 1:
                                new_pie_charts_data['color'].append(COLOR_SkyBlue)
                        elif b_DECL and num_classes == 3:
                            if j == 0:
                                new_pie_charts_data['color'].append(COLOR_Orange)
                            elif j == 1:
                                new_pie_charts_data['color'].append(COLOR_White)
                            else:
                                new_pie_charts_data['color'].append(COLOR_SkyBlue)
                        else:
                            new_pie_charts_data['color'].append(palette[j])
#            som_plot.cmap_pie = bokeh.transform.linear_cmap(field_name='value', palette=palette, low=0, high=1, high_color="#B600FF", low_color="#B600FF", nan_color="#7F7F7F")
#            som_plot.colorbar_pie = bokeh.models.ColorBar(color_mapper=som_plot.cmap_pie['transform'], ticker=bokeh.models.FixedTicker(ticks=ticks),
#                                                      major_label_overrides={tick:name for tick,name in zip(ticks, tick_names)},
#                                                      label_standoff=1, height=10, padding=-1, title="Feature", title_standoff=0)
            if b_DECL:
                new_source['color'][I0:I1] = ['#7f7f7f']*n
            else:
                new_source['color'][I0:I1] = ['#ffffff']*n
            new_source['value'][I0:I1] = np.zeros(n)
            #som_plot.hover.tooltips = HOVER_TOOLTIPS_Standard
            donutMatrix.update(som_plot_index, new_pie_charts_data)
#            p.below[som_plot_index+1] = som_plot.colorbar_pie # the som_plot_index-th colorbar should correspond to this som_plot
        else:
            donutMatrix.remove(som_plot_index)
#            p.below[som_plot_index+1] = som_plot.colorbar
    
    source.data = new_source # update everything at once
    source_pie.data = donutMatrix.create_source() # update everything at once

# Row 1
w_FigureHeight = pn.widgets.IntInput(name="plot height", width=100, value=600, start=100, end=20_000)
pn.bind(lambda height: setattr(p,'height',height), w_FigureHeight, watch=True)
w_MaxSOMWidth = pn.widgets.IntInput(name="plots/line", width=100, value=4, start=1, end=1000)
pn.bind(lambda _: update_feature_visualization(SOM_plots), w_MaxSOMWidth, watch=True)
w_mynewbutton = pn.widgets.Button(name="Change Background")
myCounterOhYeah = False
def change_background_fill_color(_):
    global myCounterOhYeah
    myCounterOhYeah = not myCounterOhYeah
    if myCounterOhYeah:
        p.background_fill_color="#ffffff"
    else:
        p.background_fill_color="#000000"
w_mynewbutton.on_click(change_background_fill_color)
w_NewSOMPlotAmount = pn.widgets.IntInput(name='# of new plots', width=110, value=1, start=1, end=30)
w_NewSOMPlotButton = pn.widgets.Button(name="Add plot")
w_NewSOMPlotButton.on_click(add_new_plot)
row1 = pn.Row(w_FigureHeight, w_MaxSOMWidth, w_mynewbutton, w_NewSOMPlotAmount, w_NewSOMPlotButton) # , w_DrawHexagonsButton

# Row 2
w_X = pn.widgets.IntInput(name='X', width=80, value=10, start=1, end=1000)
w_Y = pn.widgets.IntInput(name='Y', width=80, value=10, start=1, end=1000)
w_Init = pn.widgets.Select(name='Initialization', width=200, options=["Random", "Linear"]) # TODO: Deactivate if b_direct_traces
w_CreateNewButton = pn.widgets.Button(name="Create New")
w_CreateNewButton.on_click(create_new)
w_Sigma = pn.widgets.FloatInput(name='Sigma', width=80, value=0.5, start=0.1, end=1000, step=1)
w_TrainButton = pn.widgets.Button(name="Train")
w_TrainButton.on_click(update)
w_Sigma_start = pn.widgets.FloatInput(name='Sigma_1', width=80, value=20, start=0.1, end=1000, step=1)
w_Sigma_end = pn.widgets.FloatInput(name='Sigma_T', width=80, value=1, start=0.1, end=1000, step=1)
w_Epochs = pn.widgets.IntInput(name='T', width=80, value=10, start=2, end=1000)
w_TrainButtonMulti = pn.widgets.Button(name="Train")
w_TrainButtonMulti.on_click(update_multi)
train_error_text = pn.pane.Str()
row2 = pn.Row(w_X, w_Y, w_Init, w_CreateNewButton, w_Sigma, w_TrainButton, w_Sigma_start, w_Sigma_end, w_Epochs, w_TrainButtonMulti, train_error_text)

# Rest
SOMPage = pn.Column(row1, row2, pn.Column(), "(nothing here yet)")