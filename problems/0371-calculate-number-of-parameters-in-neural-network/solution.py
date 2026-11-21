def calculate_parameters(layers: list[dict]) -> int:
    """
    Calculate the total number of trainable parameters in a neural network.

    Args:
        layers: List of dictionaries, each describing a layer.

    Returns:
        Total number of trainable parameters (int).
    """
    s = 0

    for a in layers:
        if a['type'] == 'conv2d': 
            in_channels=a['in_channels']
            out_channels=a['out_channels']
            kernel_size=a['kernel_size']
            bias = a.get('bias', True)
            s += in_channels * out_channels * kernel_size * kernel_size
            if bias:
                s += out_channels 

        if a['type'] == 'dense':
            input_size = a['input_size']
            output_size = a['output_size']
            bias = a.get('bias', True)

            s += input_size * output_size

            if bias:
                s += output_size

    return s 
