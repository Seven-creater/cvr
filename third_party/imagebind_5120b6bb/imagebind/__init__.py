"""Pinned ImageBind model runtime used by Audio-CVR.

The upstream package imports its optional `data` module here. Audio-CVR supplies
its own PyAV preprocessing, so importing that optional pytorchvideo-dependent
module would make the model package unusable on the target server.
"""
