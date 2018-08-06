#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 15:41:34 2018

@author: Jovan Z Bebic
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go

app = dash.Dash()

df = pd.read_csv('ABB List Clean.csv')
df = df[df['Contract Year'] >= 1960]

app.layout = html.Div([    
    # column 1
    html.Div([
        html.Div([
            html.H3('ABB FACTS project references'),
            dcc.Graph(id='ABB FACTS Projects'),
        ], style={'marginBottom':'2.0em'}),
    ], style={'width':'90%', 'break-inside':'avoid'}), #'width':800

    # column 2        
    html.Div([
        html.Div([
            html.H3('Select equipment types to consider'), # 
            dcc.Checklist(id='equipments chklist',
                options=[
                    {'label': 'FSC', 'value': 'FSC'},
                    {'label': 'TCSC', 'value': 'TCSC'},
                    {'label': 'SVC', 'value': 'SVC'},
                    {'label': 'STATCOM', 'value': 'STATCOM'},
                    {'label': 'PCS6000', 'value': 'STATCOM PCS6000'},
                    {'label': 'OLC', 'value': 'OLC'},
                    {'label': 'DynaPeaQ', 'value': 'DynaPeaQ'},
                ],
                values=['FSC', 'TCSC', 'SVC', 'STATCOM', 'STATCOM PCS6000'],
                labelStyle = {'display': 'inline-block'}
            ),
        ]),
    
        html.Div([
            html.H3('Select application types to consider'),
            dcc.Checklist(id='applications chklist',
                options=[
                    {'label': 'Utility', 'value': 'Utility'},
                    {'label': 'Renewable', 'value': 'Renewable'},
                    {'label': 'Industry', 'value': 'Industry'},
                    {'label': 'Rolling Mill', 'value': 'Rolling Mill'},
                    {'label': 'Pilot', 'value': 'Pilot'},
                ],
                values=['Utility', 'Renewable'],
                labelStyle = {'display': 'inline-block'}
            ),
        ]),

        html.Div([
            html.H3('Select limits for project years'),
            dcc.RangeSlider(id='lim year slider',
                min = df['Contract Year'].min(), # 5*(int(df['Contract Year'].min()/5)),
                max = df['Contract Year'].max(), # 5*(int(df['Contract Year'].max()/5)+1),
                # included = False,
                marks = {str(year): str(year) for year in range(5*(int(df['Contract Year'].min()/5)), 5*(int(df['Contract Year'].max()/5)+1),10)},
                value = [df['Contract Year'].min(), df['Contract Year'].max()],
            ),
        ], style={'break-inside':'avoid'}),
    
        html.Div([
            html.H3('Select limits for voltage level'),
            dcc.RangeSlider(id='lim voltage slider',
                min = df['Voltage'].min(),
                max = df['Voltage'].max(),
                step = 23,
                marks = {str(vltg): str(vltg) for vltg in range(0, 23*int(df['Voltage'].max()/23 + 1), 100)},
                # marks = {np.array_str(df['Voltage'].unique())},
                # marks = {str(vltg) : str(vltg) for vltg in df['Voltage'].unique()},
                value = [69, df['Voltage'].max()],
            ),
        ], style={'break-inside':'avoid'}),
                    

    ], style={'width':'90%', 'break-inside':'avoid'}), # 'width':'300'

    # column 3
    html.Div([
        html.Div([
            html.H3('Market size', style={'marginTop':'2.0em', 'marginBottom':'0.2em'}),
            dcc.Markdown(id='market size'),
            html.Table(id='market table')
        ])

    ], style={'width':'90%', 'break-inside':'avoid'}), # 'width':500
            
], style={'marginLeft':'2%', 'marginRight':'2%', 'text-align':'left', 'columnCount':3}) # # 'marginLeft':'10%', 'marginRight':'10%', 'display':'inline-block'
                

@app.callback(dash.dependencies.Output('market size', 'children'),
             [dash.dependencies.Input('lim year slider', 'value'),
              dash.dependencies.Input('lim voltage slider', 'value')]
    )
def UpdateMarket(yrange, vrange):
    txtList = []
    txtList.append('Year range: **%d** - **%d**' %(yrange[0], yrange[1]))
    txtList.append('Voltage range: **%.1f** - **%.1f**' %(vrange[0], vrange[1]))
    market_size = '\n\n'.join(txtList)
    return market_size

@app.callback(dash.dependencies.Output('market table', 'children'),
             [dash.dependencies.Input('lim year slider', 'value'),
              dash.dependencies.Input('lim voltage slider', 'value'),
              dash.dependencies.Input('equipments chklist', 'values'),
              dash.dependencies.Input('applications chklist', 'values')]
    )
def TabulateMarket(yrange, vrange, equipments, applications):
    df1 = df[(df['Contract Year'] >= yrange[0]) & (df['Contract Year'] <= yrange[1]) & 
         (df['Voltage'] >= vrange[0]) & (df['Voltage'] <= vrange[1]) & 
         (df['Type'].isin(equipments)) &
         (df['Application'].isin(applications))]
    colTitles = ['Equipment', 'Total ratings [MVAr]', 'Retrofit [%]', 'Value [$M]']
    tblHead = [html.Tr([html.Th(col) for col in colTitles])]
    rp = {'FSC':7.0, 'TCSC':3.0, 'SVC':12.0, 'STATCOM':15.0, 'STATCOM PCS6000':10.3, 'OLC':22.1, 'DynaPeaQ':10.1}
    tblBody = []
    for equipment in equipments:
        total = df1[df1['Type'] == equipment]['Capacitive Rating'].sum()
        rpval = rp[equipment] if equipment in rp.keys() else 0.0
        tblVals = [equipment, '{0:,.0f}'.format(total), '%.1f'%(rpval), '{0:,.0f}'.format(total*rpval/100*0.15)]
        tblBody.append(html.Tr([html.Td(col) for col in tblVals]))
    
    return tblHead + tblBody


@app.callback(dash.dependencies.Output('ABB FACTS Projects', 'figure'),
             [dash.dependencies.Input('lim year slider', 'value'),
              dash.dependencies.Input('lim voltage slider', 'value'),
              dash.dependencies.Input('equipments chklist', 'values'),
              dash.dependencies.Input('applications chklist', 'values')]
    )
def UpdateFigure(yrange, vrange, equipments, applications):
    df1 = df[(df['Contract Year'] >= yrange[0]) & (df['Contract Year'] <= yrange[1]) & 
             (df['Voltage'] >= vrange[0]) & (df['Voltage'] <= vrange[1]) & 
             (df['Type'].isin(equipments)) &
             (df['Application'].isin(applications))] # df1 = df[(df['Contract Year'] >= lowYear) & (df['Contract Year'] <= hiYear)]
    return {            
        'data': [
            go.Scatter(
                x=df1[df1['Type'] == i]['Contract Year'],
                y=df1[df1['Type'] == i]['Capacitive Rating'],
                text= df1[df1['Type'] == i]['End Country']+', ' + df1[df1['Type'] == i]['Client']+'<BR>'+
                      df1[df1['Type'] == i]['Project'], # +'<BR>'+ 
                      # '('+df[df['Type'] == i]['Capacitive Rating']+')', 
                      # '('+str(-df[df['Type'] == i]['Inductive Rating'])+', '+str(df[df['Type'] == i]['Capacitive Rating'])+')',
                mode='markers',
                opacity=0.7,
                marker={
                    'size': 15., # 15.*df['Capacitive Rating']/100.
                    'line': {'width': 0.5, 'color': 'white'}
                },
                hoverinfo='x+text',
                name=i
            ) for i in df['Type'].unique()
        ],
        'layout': go.Layout(
            xaxis={'title': 'Project Year'}, # 'range': [1960, 2020]
            yaxis={'title': 'Capacitive Ratings [MVAr]'},
            margin={'l': 60, 'b': 40, 't': 10, 'r': 10},
            legend={'x': 0, 'y': 1},
            hovermode='closest',
        )
    }

if __name__ == '__main__':
    app.run_server()