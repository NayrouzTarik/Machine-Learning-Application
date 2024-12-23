from dash import Dash, html, dcc, Input, Output
from django_plotly_dash import DjangoDash
import plotly.graph_objs as go

def create_dash_app():
    app = DjangoDash('ModelVisualization')
    
    app.layout = html.Div([
        dcc.Graph(id='metrics-chart'),
        dcc.Store(id='model-results')
    ])

    @app.callback(
        Output('metrics-chart', 'figure'),
        Input('model-results', 'data')
    )
    def update_graph(results):
        if not results:
            return {}

        if 'accuracy' in results:
            return create_classification_chart(results)
        elif 'train_score' in results:
            return create_performance_chart(results)
        elif 'silhouette' in results:
            return create_clustering_chart(results)
        
        return {}

    return app

def create_classification_chart(results):
    return {
        'data': [go.Bar(
            x=['Accuracy', 'Precision', 'Recall', 'F1'],
            y=[results['accuracy'], results['precision'], 
               results['recall'], results['f1']],
            marker_color='rgb(255, 105, 180)'
        )],
        'layout': get_layout('Classification Metrics')
    }

def get_layout(title):
    return {
        'title': title,
        'template': 'plotly_dark',
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'font': {'color': '#fff'}
    }

app = create_dash_app()