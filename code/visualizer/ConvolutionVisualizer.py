import sys, os

visualization_source_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, "pytorch-cnn-visualizations", "src"))
sys.path.append(visualization_source_dir)
import cnn_layer_visualization

class ConvolutionVisualizer(object):
    def __init__(self, options):
        self.options = options
    
    def visualize(self, model, selected_layer, output_dir):
        viz = cnn_layer_visualization.CNNLayerVisualization(model.features, selected_layer, 0)
        viz.visualise_layer_with_hooks(output_dir) 
