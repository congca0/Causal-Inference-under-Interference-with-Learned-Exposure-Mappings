library(DiagrammeR)

grViz("
digraph summary {

graph[
layout = dot,
rankdir = LR,
fontsize = 16
]

node[
shape=box,
style='rounded,filled',
fillcolor='#F7F7F7',
color='#666666',
fontname=Helvetica,
fontsize=14,
margin='0.25,0.15'
]

P[
label='Similar Pollution\nPrediction'
]

T[
label='Different\nTransport Recovery'
]

E[
label='Different Exposure\nMappings'
]

S[
label='Different Spillover\nConclusions'
]

P -> T -> E -> S

node[
shape=note,
style='filled',
fillcolor='#EAF4FC',
fontsize=12
]

PDE[
label='PDE'
]

PINO[
label='PINO'
]

FNO[
label='FNO'
]

GeoPT[
label='GeoPT'
]

PDE -> T [style=dotted,arrowhead=none]
PINO -> T [style=dotted,arrowhead=none]
FNO -> T [style=dotted,arrowhead=none]
GeoPT -> T [style=dotted,arrowhead=none]

}
")