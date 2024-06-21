import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
import plotly.graph_objects as go
import plotly.express as px

df_33 = pd.read_excel("/Users/sarthakmishra/Documents/Code/Lechler_3D_Nozzel/results/33.xlsx")
df_111 = pd.read_excel("/Users/sarthakmishra/Documents/Code/Lechler_3D_Nozzel/results/111.xlsx")
num_of_distributions = 20 # Number of distributions to plot

def csv_to_discrete_distribution(num_of_distributions, df, colorscale='Jet'):
    # Step 1: Duplicate df_111 num_of_distributions times
    df_duplicated = pd.concat([df]*num_of_distributions, ignore_index=True)
    
    df_duplicated['y'] = np.repeat(np.arange(1, num_of_distributions+1), len(df))
    
    # Create a scatter plot
    scatter = go.Scatter3d(
        x=df_duplicated['xpos'],
        y=df_duplicated['y'],
        z=df_duplicated['level'],
        mode='markers',
        marker=dict(
            size=6,
            color=df_duplicated['level'],
            colorscale=colorscale,
            showscale=True
        )
    )
    
    # Create a figure and add the scatter plot
    fig = go.Figure(data=[scatter])
    
    return fig
    


def csv_to_3D_Distribution(num_of_distributions, df, colorscale='Jet'):
    # Step 1: Duplicate df_111 num_of_distributions times
    df_duplicated = pd.concat([df]*num_of_distributions, ignore_index=True)

    df_duplicated['y'] = np.repeat(np.arange(1, num_of_distributions+1), len(df))


    # Assume df_duplicated has columns 'xpos', 'y', and 'level'
    X = df_duplicated['xpos'].values
    Y = df_duplicated['y'].values
    Z = df_duplicated['level'].values.reshape(-1, len(np.unique(X)))

    # Create a grid of X and Y values
    X, Y = np.meshgrid(np.unique(X), np.unique(Y))

    # Create a 3D surface plot
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale=colorscale, colorbar=dict(title='level (ml)'))])
    
    fig.update_layout(
    title='Lechler Nozzel Distribution',
    scene=dict(
        xaxis_title='xpos',
        yaxis_title='ypos',
        zaxis_title='level (ml)'
    )
    
    )
    
    return fig 

def csv_to_2D_Heatmap(df, colorscale='Jet'):
    # Step 1: Duplicate df_111 num_of_distributions times
    df_duplicated = pd.concat([df]*num_of_distributions, ignore_index=True)

    df_duplicated['y'] = np.repeat(np.arange(1, num_of_distributions+1), len(df))

    # Create a heatmap
    heatmap = go.Heatmap(
        x=df_duplicated['xpos'],
        y=df_duplicated['y'],
        z=df_duplicated['level'],
        colorscale=colorscale
    )

    # Create a figure and add the heatmap
    fig = go.Figure(data=[heatmap])

    # Set layout properties
    fig.update_layout(
        title='Heatmap of Z values',
        xaxis_title='X',
        yaxis_title='Y',
    )

    return fig


