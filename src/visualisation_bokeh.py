import pandas as pd
import numpy as np
import os

#from bkcharts import Chord
from bkcharts import Chord, Donut
from bokeh.io import show, output_file, export_png
from bokeh.plotting import figure


def renderChordChart(df, dfSource, dfTarget, dfValue, dfTitle=''):
    chord_from_df = Chord(df, 
        source=dfSource, target=dfTarget, value=dfValue, 
        title=dfTitle)
    chord_from_df.plot_height = 1600
    chord_from_df.plot_width  = 1600
    output_file('tempChordChart.html')
    show(chord_from_df)
    return chord_from_df;

def renderDonutChart(df, labels, values, textFontSize, hoverText, dfTitle=''):
    donutChart = Donut(df, label=labels, values=values,
        text_font_size=textFontSize, hover_text=hoverText,
        title=dfTitle)
    donutChart.plot_height = 1600
    donutChart.plot_width  = 1600
    output_file('tempDonutChart.html')
    show(donutChart)
    return donutChart;


def renderHistogram(histogram, edges, figTitle, xLabel, yLabel):
    p1 = figure(title=figTitle, tools="save",
                background_fill_color="#353030")

    p1.quad(top=histogram, bottom=0, left=edges[:-1], right=edges[1:],
            fill_color="#FFA732", line_color="#E2DDD7")

    p1.plot_width  = 800

    p1.legend.location = "center_right"
    p1.legend.background_fill_color = "darkgrey"
    p1.xaxis.axis_label = xLabel
    p1.yaxis.axis_label = yLabel

    output_file('tempHistogram.html')
    show(p1)
    return p1;



def exportFigure(bkFig, figPath='tester.png'):
    export_png(bkFig, figPath)