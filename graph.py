#!/usr/bin/env python3
import os
from pathlib import Path
import plotly.graph_objects as go
from cairosvg import svg2png

INPUT='input'
WIDTH = 3*300
HEIGHT = 2*300
SCALE = 2

LEGEND_Y_POSITION= 0.9
LEGEND_X_POSITION = 1
LEGEND_FONT_SIZE = 12

X_LABELS = ["2048", "16384", "65536"]

CUDA_BASIC = [0.3992664, 0.773925, 5.087871]
CUDA_SHARED = [0.5498873999999999, 0.5966728, 6.2167639999999995]
CPU = [0.0, 7.1875, 43.0571426]

Y_LABELS  = ["Cuda Basic", "Cuda Shared", "CPU"]

def make_png():
  p = Path('images')
  for i in p.glob('**/*'):
      if (i.as_posix().endswith('.svg')):
        with open(i.as_posix(), "rb") as file:
          data= file.read()
          svg2png(bytestring=data,write_to=f"images/{i.stem}.png")
        os.remove(i.as_posix())

def graph(labels, data, name) -> None:
    fig = go.Figure()
    for label, y_data in zip(labels, data):

        xarr = X_LABELS
        yarr = y_data

        fig.add_trace(go.Scatter(
            x=xarr,
            y=yarr,
            name= label,
            connectgaps=True,
            mode='lines'
        ))

    fig.update_layout(
                        xaxis_title='Number of Input Points',
                        yaxis_title='Execution Time (ms)',
                        title="Number of Inputs vs Execution Time",
                )
    fig.update_layout(legend=dict(y=LEGEND_Y_POSITION,x=LEGEND_X_POSITION, traceorder='reversed', font_size=LEGEND_FONT_SIZE))

    fig.write_image(f"images/{name}.svg", width=WIDTH, height=HEIGHT, scale = SCALE)

if __name__ == "__main__":
    graph(Y_LABELS, [CUDA_BASIC, CUDA_SHARED, CPU], "timing_difference")
    graph(["Cuda Basic", "Cuda Shared"], [CUDA_BASIC, CUDA_SHARED], "timing_difference_GPU")
    make_png()
    print("DONE")
