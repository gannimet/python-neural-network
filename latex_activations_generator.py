import numpy

layer_offset = 0
hspace = 5
positive_color = 'ProcessBlue'
negative_color = 'OrangeRed'
run_suffix = 3

for predicted_number in range(10):
  activations = numpy.load(f"./trained_activations/activations_{predicted_number}.npz")
  activations = [activations[key] for key in activations]
  hidden_layer_max = max(numpy.max(activations[1]), numpy.max(activations[2]), numpy.max(activations[3]))
  tex_filename = f"./trained_activations/mnist_{predicted_number}_{run_suffix}.tex"

  def get_neuron_color(neuron_activation, max_value=1):
    neuron_color = positive_color if neuron_activation >= 0 else negative_color
    ratio = neuron_activation / max_value
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
        f.write(f"\\node[neuron, fill={get_neuron_color(neuron_activation, hidden_layer_max)}] (H{l}_{i}) at ({layer_offset + l*hspace}cm, {16-i}cm) {{}};\n")
      
    # Output layer
    for i in range(10):
      neuron_activation = activations[4][i, 0]
      f.write(f"\\node[neuron, fill={get_neuron_color(neuron_activation)}] (O_{i}) at ({layer_offset + hspace*3}cm, {12.5 - i}cm) {{${i}$}};\n")
      
    suffix = r"""\foreach \source in {0,...,16}
          \foreach \dest in {1,...,16}
            \path (H0_\source) edge[-] (H1_\dest);

        \foreach \source in {0,...,16}
          \foreach \dest in {1,...,16}
            \path (H1_\source) edge[-] (H2_\dest);

        \foreach \source in {0,...,16}
          \foreach \dest in {0,...,9}
            \path (H2_\source) edge[-] (O_\dest);
    \end{tikzpicture}

    \end{document}"""
    f.write(suffix)