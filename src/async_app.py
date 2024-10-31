"""
This module contains async application to display hand movements in realtime.

Author: Ivan Khrop
Date: 14.09.2024
"""

# Import necessary modules
import dash
from time import sleep
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from threading import Event, Barrier
import warnings

# Own imports
from mediapipe import solutions
from camera_thread.rs_thread import CameraThreadRS
from camera_thread.processing_thread import FusionThread
from utils.fusion import DataMerger
from utils.constants import TIME_DELTA

# get all cameras we have
available_cameras = CameraThreadRS.returnCameraIndexes()

assert len(available_cameras) > 0, "Please connect a camera !"

# Set up event and threads
close_threads = Event()  # to close threads
camera_barrier = Barrier(parties=len(available_cameras))  # just all cameras
data_barrier = Barrier(
    parties=(len(available_cameras) + 1)
)  # all cameras + one processing thread !!!
read_finished = Barrier(
    parties=(len(available_cameras) + 1)
)  # all cameras + one processing thread !!!

# Camera threads initialization
threads = {
    camera_id: CameraThreadRS(
        camera_name=camera_name,
        camera_id=camera_id,
        close_event=close_threads,
        barrier=camera_barrier,
        data_barrier=data_barrier,
    )
    for camera_name, camera_id in available_cameras
}

data_merger = DataMerger(time_delta=TIME_DELTA)
fusion_thread = FusionThread(
    stop_thread=close_threads,
    sources=threads,
    merger=data_merger,
    data_barrier=data_barrier,
)

# Dash app initialization
app = dash.Dash(__name__)
# do now show logging data
app.enable_dev_tools(dev_tools_silence_routes_logging=True)

# Layout
app.layout = html.Div(
    [
        html.H1("Real-Time Plotly Hand Position", style={"textAlign": "center"}),
        html.Div(
            dcc.Graph(id="3d-plot"),
            style={
                "display": "flex",  # Flexbox layout
                "justify-content": "center",  # Center horizontally
                "align-items": "center",  # Center vertically
            },
        ),
        dcc.Interval(id="interval-component", interval=125, n_intervals=0),
        html.Button("Start Threads", id="start-button", n_clicks=0),
        html.Button("Stop Threads", id="stop-button", n_clicks=0),
    ]
)

# define layout for plotly
custom_layout = go.Layout(
    autosize=False,
    width=800,
    height=600,
    showlegend=False,
    scene=dict(
        xaxis_title="X Axis",
        yaxis_title="Y Axis",
        zaxis_title="Z Axis",
        xaxis=dict(range=(0.0, 2.0), autorange=False),  # Set the x-axis limit
        yaxis=dict(range=(-1.0, 1.0), autorange=False),  # Set the y-axis limit
        zaxis=dict(range=(0.0, 1.0), autorange=False),  # Set the z-axis limit
        camera=dict(eye=dict(x=1.0, y=1.0, z=1.0)),
        aspectmode="manual",  # Fixes the aspect ratio
        aspectratio=dict(x=3.0, y=3.0, z=1),  # Ensures aspect ratio remains constant
    ),
    margin=dict(l=0, r=0, t=0, b=0),  # Tight margins for better visualization
)


# Callbacks for starting/stopping threads
@app.callback(Output("start-button", "n_clicks"), [Input("start-button", "n_clicks")])
def start_threads(n_clicks):
    if n_clicks > 0:
        for camera_id in threads:
            threads[camera_id].start()
        fusion_thread.start()
    return n_clicks


@app.callback(Output("stop-button", "n_clicks"), [Input("stop-button", "n_clicks")])
def stop_threads(n_clicks):
    if n_clicks > 0:
        close_threads.set()
        fusion_thread.join()
    return n_clicks


# Callback for updating the graph
@app.callback(Output("3d-plot", "figure"), [Input("interval-component", "n_intervals")])
def update_graph_live(n_intervals: int):
    _, hands = data_merger.get_latest_result()
    if hands is None:
        return go.Figure(layout=custom_layout)

    fig = go.Figure(layout=custom_layout)

    for hand in hands:
        landmarks = hands[hand]
        scatter_data = go.Scatter3d(
            x=landmarks.loc[:].x.values,
            y=landmarks.loc[:].y.values,
            z=landmarks.loc[:].z.values,
            mode="markers+text",
            text=[str(idx) for idx in landmarks.index],
            marker=dict(size=3, color="black"),
            textfont=dict(size=6, color="blue"),
        )
        fig.add_trace(scatter_data)

        connections_x, connections_y, connections_z = [], [], []
        for start_idx, end_idx in solutions.hands.HAND_CONNECTIONS:
            connections_x += [
                landmarks.loc[start_idx].x,
                landmarks.loc[end_idx].x,
                None,
            ]
            connections_y += [
                landmarks.loc[start_idx].y,
                landmarks.loc[end_idx].y,
                None,
            ]
            connections_z += [
                landmarks.loc[start_idx].z,
                landmarks.loc[end_idx].z,
                None,
            ]

        fig.add_trace(
            go.Scatter3d(
                x=connections_x,
                y=connections_y,
                z=connections_z,
                mode="lines",
                line=dict(color="black", width=2),
            )
        )

    return fig


if __name__ == "__main__":
    # get rid of warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    # sleep a bit to configure mediapipe
    sleep(2.0)
    print("\nStart Application\n")

    # run application
    app.run_server(debug=False, use_reloader=False)
