import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np 

def read_log_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    epochs = []
    duration_loss = []
    prior_loss = []
    diffusion_loss = []

    for line in lines:
        if line.startswith('Epoch'):
            parts = line.split(' | ')
            epoch_info = parts[0].split(':')
            epochs.append(int(epoch_info[0].split()[1]))
            duration_loss.append(float(parts[0].split('=')[1].strip()))
            prior_loss.append(float(parts[1].split('=')[1].strip()))
            diffusion_loss.append(float(parts[2].split('=')[1].strip()))

    return epochs, duration_loss, prior_loss, diffusion_loss

# Read data from log files
train_epochs, train_duration_loss, train_prior_loss, train_diffusion_loss = read_log_file('./logs/final_exp/7/train.log')
test_epochs, test_duration_loss, test_prior_loss, test_diffusion_loss = read_log_file('./logs/final_exp/7/test.log')

print("epochs: " , train_epochs, "train_duration_loss:" , train_duration_loss, "train_diffusion_loss:" , train_diffusion_loss)
# Create subplots
fig = make_subplots(rows=1, cols=2, subplot_titles=['Duration Loss', 'Diffusion Loss'])

# Add training loss traces for Duration Loss subplot
fig.add_trace(go.Scatter(x=train_epochs, y=train_duration_loss, mode='lines', name='Train Duration Loss'), row=1, col=1)

# Add testing loss traces for Duration Loss subplot
fig.add_trace(go.Scatter(x=test_epochs, y=test_duration_loss, mode='lines', name='Test Duration Loss'), row=1, col=1)

# Add training loss traces for Diffusion Loss subplot
fig.add_trace(go.Scatter(x=train_epochs, y=train_diffusion_loss, mode='lines', name='Train Diffusion Loss'), row=1, col=2)

# Add testing loss traces for Diffusion Loss subplot
fig.add_trace(go.Scatter(x=test_epochs, y=test_diffusion_loss, mode='lines', name='Test Diffusion Loss'), row=1, col=2)

# Update layout
fig.update_layout(title='Training and Testing Losses',
                  legend=dict(x=0, y=1, traceorder='normal'))

# Show plot
fig.show()
