"""Hosted anatomical mesh viewer, without a visitor-local asset server."""
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
import streamlit as st

@st.cache_data(show_spinner=False)
def load_mesh(path):
    vertices, faces = [], []
    for line in Path(path).read_text(errors="replace").splitlines():
        if line.startswith("v "):
            vertices.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            ids = [int(x.split("/")[0])-1 for x in line.split()[1:]]
            for n in range(1, len(ids)-1):
                faces.append([ids[0], ids[n], ids[n+1]])
    return np.asarray(vertices), np.asarray(faces)

def render_heart_viewer(asset_dir):
    st.title("Heart anatomy")
    st.write("Rotate and zoom the anatomical mesh. This is an educational illustration, not a scan or a model of your heart.")
    model = st.radio("View", ["External", "Open heart"], horizontal=True)
    filename = "heart1.obj" if model == "External" else "openheartLD1.obj"
    with st.spinner("Loading anatomy..."):
        v, f = load_mesh(str(Path(asset_dir)/"models"/filename))
    mesh = go.Mesh3d(x=v[:,0], y=v[:,1], z=v[:,2], i=f[:,0], j=f[:,1], k=f[:,2],
                     color="#bd6972", flatshading=False, hoverinfo="skip",
                     lighting=dict(ambient=.5, diffuse=.8, specular=.25, roughness=.7))
    fig = go.Figure(mesh)
    axis = dict(visible=False)
    fig.update_layout(height=610, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="#121b29",
                      scene=dict(xaxis=axis,yaxis=axis,zaxis=axis,aspectmode="data",bgcolor="#121b29"))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
    st.caption("Drag to rotate · Scroll or pinch to zoom · Geometry from the repository's existing OBJ assets. Disease progression and blood-flow simulations are not validated and are not shown.")
