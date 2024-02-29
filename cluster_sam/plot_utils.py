from typing import Dict, Any
import cv2
from plotly import graph_objects as go
import plotly.express as px

from models import CV2Image, SAMMasksInFace


PLOTLY_CONFIG:Dict[str, Any] = {
    "displaylogo": False,
    "modeBarButtonsToAdd": [
        "drawline",
        "drawopenpath",
        "drawclosedpath",
        "drawcircle",
        "drawrect",
        "eraseshape",
        "autoScale2d",
        "toggleSpikelines",
    ],
    "modeBarButtonsToRemove": [
        "select2d",
        "lasso2d",
        "hoverClosestCartesian",
        "hoverCompareCartesian",
    ],
}


def plot_cluster(
    data: SAMMasksInFace,
    img: CV2Image = None,
    title: str = None,
    plot_img: bool = False,
    plot_centroids: bool = True,
    class_colors: dict = None,
    fill_masks: bool = True,
) -> go.Figure:
    fig = go.Figure()
    class_colors = class_colors or {}

    if plot_img is True and data.original_image_path is not None:
        if img is None:
            img = cv2.imread(str(data.original_image_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        b = data.roi_box
        img = img[b.y_min : b.y_min + b.height, b.x_min : b.x_min + b.width]
        fig = px.imshow(img)

    fig.update_layout(
        title=title,
        width=1600,
        height=1000,
    )
    fig.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1)

    for i, mask in enumerate(data.masks):
        name = f"{str(mask.label).title()} {i+1}"
        text = ""
        if mask.area_normalized is not None:
            text += f"<b>Area Normalized:</b> {mask.area_normalized:.3f}<br>"
        if mask.iou is not None:
            text += f"<b>IoU:</b> {mask.iou:.3f}<br>"
        _, _, w, h = mask.bbox
        text += f"<b>Aspect Ratio:</b> {w/h:.3f}<br>"
        mask_to_bbox_ratio = mask.area / (w*h)
        text += f"<b>Mask to BBox Ratio:</b> {mask_to_bbox_ratio:.3f}"
        mask.segmentation.plot(name=name, text=text, fig=fig, color=class_colors.get(mask.label), fill=fill_masks)

    if plot_centroids:
        markers_labels, markers_x, markers_y = [], [], []
        for i, mask in enumerate(data.masks):
            markers_labels.append(f"<b>{str(i+1)}</b>")
            markers_x.append(mask.segmentation.centroid[0])
            markers_y.append(mask.segmentation.centroid[1])
        fig.add_trace(
            go.Scatter(
                x=markers_x,
                y=markers_y,
                mode="markers+text",
                name="Centroids",
                marker=dict(
                    size=8,
                    symbol="star",
                    color="yellow",
                    line=dict(width=1, color="black"),
                ),
                text=markers_labels,
                textposition="bottom center",
                textfont=dict(color="black", size=8),
                hoverinfo="x+y",
            )
        )

    return fig


def plot_cluster_binary(
        data: SAMMasksInFace,
        title: str = None,
        plot_binary: bool = True,
        plot_projections: bool = True,
        plot_contours: bool = False,
        plot_bbox: bool = False,
        fig_height: int = 900,
        fig_width: int = 700,
        plot_bgcolor:str = "white",
        font_size:int = 12,
        font_family:str = "Sans Serif, monospace",
        auto_range:bool = False,
        ) -> go.Figure:
    if not any([plot_binary, plot_contours]):
        raise ValueError("At least one of plot_binary or plot_contours must be True")
    if len(data.masks) == 0:
        raise ValueError("No masks to plot")

    # Merge all masks into one to create a cluster
    segments = [mask.segmentation for mask in data.masks]
    cluster = sum(segments[1:], segments[0]) # This is possible because of the __add__ method in Segmentation

    if plot_binary:
        fig = cluster.plot_binary(plot_bbox=plot_bbox)
    else:
        fig = go.Figure()
        fig.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1)

    if plot_contours:
        fig = cluster.plot(fig=fig, mode='lines', fill=False, plot_bbox=plot_bbox, plot_centroid=True)

    # Set Layout
    fig.update_layout(
        title=title,
        coloraxis_showscale=False,
        xaxis=dict(
            zeroline=True,
            showgrid=False,
            ticks="inside",
            showline=True,
            mirror='all',
            linewidth=1,
            linecolor="black",
        ),
        yaxis=dict(
            zeroline=True,
            showgrid=False,
            ticks="inside",
            showline=True,
            mirror='all',
            linewidth=1,
            linecolor="black",
        ),
        height=fig_height,
        width=fig_width,
        bargap=0,
        hovermode="closest",
        showlegend=False,
        xaxis_title="X",
        yaxis_title="Y",
        font=dict(
            size=font_size,
            family=font_family,
        ),
        plot_bgcolor=plot_bgcolor,
        dragmode="zoom",
    )

    if plot_projections:
        # Marginal X
        fig.add_trace(
            go.Scatter(
                xaxis="x2",
                x=cluster.binary_mask.sum(axis=1),  # Vertical projection
                fill="tozeroy",
                name="h_projection",
                hoverinfo="x+y+name",
                line=dict(color="blue"),
            )
        )
        # Marginal Y
        fig.add_trace(
            go.Scatter(
                yaxis="y2",
                y=cluster.binary_mask.sum(axis=0),  # Horizontal projection
                fill="tozeroy",
                name="v_projection",
                hoverinfo="x+y+name",
                line=dict(color="green"),
            )
        )
        fig.update_layout(
            xaxis=dict(
                zeroline=True,
                domain=[0, 0.85],
                showgrid=False,
                showspikes=True,
                spikemode="across",
                spikethickness=1,
                ticks="inside",
                showline=True,
                mirror='all',
                linewidth=1,
                linecolor="black",
            ),
            yaxis=dict(
                zeroline=True,
                domain=[0, 0.85],
                showgrid=False,
                showspikes=True,
                spikemode="across",
                spikethickness=1,
                ticks="inside",
                showline=True,
                mirror='all',
                linewidth=1,
                linecolor="black",
            ),
            xaxis2=dict(
                zeroline=True,
                domain=[0.85, 1],
                showgrid=False,
                showspikes=True,
                spikemode="across",
                spikethickness=1,
                ticks="inside",
                showline=True,
                mirror='all',
                linewidth=1,
                linecolor="black",
            ),
            yaxis2=dict(
                zeroline=True,
                domain=[0.9, 1],
                showgrid=False,
                showspikes=True,
                spikemode="across",
                spikethickness=1,
                ticks="inside",
                showline=True,
                mirror='all',
                linewidth=1,
                linecolor="black",
                anchor="x1",
                side="left",
            ),
            xaxis2_title="Count",
            yaxis2_title="Count",
        )
    if auto_range:
        b = cluster.bbox
        fig.update_layout(
            # xaxis=dict(range=[b.x_min, b.x_max]),
            yaxis=dict(range=[b.y_min, b.y_max])
            )
    return fig
