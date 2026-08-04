library(DiagrammeR)

grViz("
digraph framework {

graph [
layout = dot,
rankdir = TB,
nodesep = 0.35,
ranksep = 0.55,
fontsize = 16
]

node [
shape = box,
style = 'rounded,filled',
fillcolor = '#F8F8F8',
color = '#666666',
fontname = Helvetica,
fontsize = 14,
margin = '0.25,0.15'
]

A [
label='Pollution-control\nIntervention (A)'
]

PM [
label='Observed Pollution Field\nPM(x,t)'
]

M [
label=<
<B>Candidate Transport Models</B><BR/><BR/>
Mechanistic: PDE<BR/>
Operator Learning: PINO, FNO<BR/>
Foundation Model: GeoPT
>
]

T [
label='Estimated Transport\nOperator T'
]

E [
label='Exposure Mapping\nE(T)'
]

Y [
label='Spillover and\nIntervention Effects'
]

A -> PM
PM -> M
M -> T
T -> E
E -> Y

node[
shape=box,
style='rounded,filled',
fillcolor='#FDECEC',
color='#C0392B',
fontcolor='#8E2A2A'
]

U[
label=<
<B>Uncertainty Propagation</B><BR/><BR/>
Transport model uncertainty<BR/>
&#8595;<BR/>
Exposure mapping uncertainty<BR/>
&#8595;<BR/>
Spillover inference uncertainty
>
]

U -> T[
style=dashed,
color='#C0392B',
arrowhead=open
]

}
")