import numpy

layer_offset = 0
hspace = 6.5
positive_color = 'ProcessBlue'
negative_color = 'OrangeRed'

for predicted_number in range(10):
  network_arrays = numpy.load(f"./trained_activations/activations_{predicted_number}.npz")
  network_arrays = [network_arrays[key] for key in network_arrays]
  activations = network_arrays[:5]
  weights = network_arrays[5:]
  hidden_layer_max = max(numpy.max(activations[1]), numpy.max(activations[2]), numpy.max(activations[3]))
  tex_filename = f"./trained_activations/mnist_{predicted_number}.tex"

  def get_scaled_color(value, max_value=1):
    neuron_color = positive_color if value >= 0 else negative_color
    ratio = value / max_value
    color_intensity = round(abs(ratio) * 100.0)
    return f"{neuron_color}!{color_intensity}"

  with open(tex_filename, "w", encoding="utf-8") as f:
    # Prefix
    prefix = r"""\documentclass[tikz,dvipsnames]{standalone}
\usepackage{tikz}
\usepackage{amsmath}
\usepackage{pgffor}

\usetikzlibrary{shapes,arrows,chains}

\begin{document}
\pagestyle{empty}

\begin{tikzpicture}[shorten >=0pt,draw=black!50]
\tikzstyle{neuron}=[circle, draw=black!100, fill=OrangeRed!100, minimum size=0.8cm,inner sep=1pt]
"""
    f.write(prefix)
    
    # Hidden layers
    for l in range(3):
      for i in range(17):
        neuron_activation = activations[l+1][i, 0]
        f.write(f"\t\\node[neuron, fill={get_scaled_color(neuron_activation, hidden_layer_max)}] (H{l}_{i}) at ({layer_offset + l*hspace}cm, {16-i}cm) {{}};\n")
      f.write("\n")
      
    # Output layer
    for i in range(10):
      neuron_activation = activations[4][i, 0]
      f.write(f"\t\\node[neuron, fill={get_scaled_color(neuron_activation)}] (O_{i}) at ({layer_offset + hspace*3}cm, {12.5 - i}cm) {{${i}$}};\n")
      
    f.write("\n")
    
    # Hidden weights
    for i in range(2, len(weights) - 1):
      matrix = weights[i]
      
      for row in range(len(matrix)):
          for col in range(len(matrix[row])):
            f.write(f"\t\\path (H{i-2}_{col}) edge[-, draw={get_scaled_color(matrix[row, col])}] (H{i-1}_{row + 1});\n")
          f.write("\n")
          
    # Output weights
    output_weights = weights[-1]
    
    for row in range(len(output_weights)):
      for col in range(len(output_weights[row])):
        f.write(f"\t\\path (H2_{col}) edge[-, draw={get_scaled_color(output_weights[row, col])}] (O_{row});\n")
      f.write("\n")
    
    # suffix = r"""\foreach \source in {0,...,16}
    #       \foreach \dest in {1,...,16}
    #         \path (H0_\source) edge[-] (H1_\dest);

    #     \foreach \source in {0,...,16}
    #       \foreach \dest in {1,...,16}
    #         \path (H1_\source) edge[-] (H2_\dest);

    #     \foreach \source in {0,...,16}
    #       \foreach \dest in {0,...,9}
    #         \path (H2_\source) edge[-] (O_\dest);
    # \end{tikzpicture}

    # \end{document}"""
    # f.write(suffix)
    
    f.write("\\end{tikzpicture}\n")
    f.write("\\end{document}")