import plotly.graph_objects as go

# Initialize lists to store data
n_values = []
f1_scores = []
precisions = []
recalls = []

# Read the log file
filename_in = '/work/alex.unicef/feature_extractor/bbox_hypothesis_experiments/experiments/metrics_1_cross_val_bbox.log'
filename_out = 'experiment_v1_bbox.html'
with open(filename_in, 'r') as file:
    lines = file.readlines()

# Parse the lines and extract data
for i in range(0, len(lines), 4):
    n_values.append(float(lines[i].split('=')[1].strip()))
    f1_scores.append(float(lines[i+1].split('=')[1].strip()))
    precisions.append(float(lines[i+2].split('=')[1].strip()))
    recalls.append(float(lines[i+3].split('=')[1].strip()))

# Create the Plotly figure
fig = go.Figure()

# Add traces with lines and markers
fig.add_trace(go.Scatter(x=n_values, y=f1_scores, mode='lines+markers', name='F1 Score'))
fig.add_trace(go.Scatter(x=n_values, y=precisions, mode='lines+markers', name='Precision'))
fig.add_trace(go.Scatter(x=n_values, y=recalls, mode='lines+markers', name='Recall'))

# Update layout
fig.update_layout(title='bbox', xaxis_title='n', yaxis_title='Metrics')
fig.show()

# Save the plot as an HTML file
fig.write_html(f'/work/alex.unicef/feature_extractor/bbox_hypothesis_experiments/experiment_visualization/{filename_out}', auto_open=True)
