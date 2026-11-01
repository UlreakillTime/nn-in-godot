extends NN
class_name CPU_NN

## Go use GPU_NN. It's way faster. Unless u are SignalRGB and want to cook 🥩 on your CPU or really wanna create AIs with this plugin while it's still not done, then fine, use CPU_NN

var layers: Array[Array]

func setup(LAYERS_SIZES: Array[int], activation_types: Array[String]) -> void:
	#const LAYERS_SIZES: Array = [100, 10, 3]
	#var start_layer: Sprite2D = find_child("Layer1")
	var iter: int = 0
	var NN: Array[Array]
	for layer_size in LAYERS_SIZES:
		var layer: Array[Array]
		if iter == 0:
			for i in range(layer_size):
				layer.append(PackedFloat32Array([0.0, 0.0]))
			#print("Starter layer, "+str([layer_size])+" neurons total")
		else:
			for i in range(layer_size):
				var neuron: PackedFloat32Array
				for b in range(LAYERS_SIZES[iter-1]+3):
					neuron.append(randf_range(0.0, 5.0/layer_size))
				layer.append(neuron)
		NN.append(layer)
		iter += 1
	var activations: Array[Callable]
	for activation in activation_types:
		activations.append(Callable(self, activation))
	NN.append(activations)
	layers = NN.duplicate(true)

func calculate_output(input: Array[float]) -> Array[float]:
	var output: Array[float]
	for neuron in calculated_nn(input, layers)[-2]:
		output.append(neuron[0])
	return output

func calculate(input: Array[float]):
	for layer_num in range(layers.size()-1):
		if layer_num == 0:
			if input.size() == layers[0].size():
				layers[0] = []
				for i in input:
					layers[0].append([i, layers[-1][0].call(i)])
			else:
				return layers[-1]
		else:
			for neuron in layers[layer_num]:
				var v: float = 0 #Value
				for prev_n_num in range(layers[layer_num-1].size()):
					v += layers[layer_num-1][prev_n_num][1] * neuron[prev_n_num+2]
				v += neuron[-1]
				neuron[0] = v
				v = layers[-1][layer_num].call(v)
				neuron[1] = v

func calculated_nn(input: Array[float], nn: Array[Array]) -> Array[Array]:
	for layer_num in range(nn.size()-1):
		if layer_num == 0:
			if input.size() == nn[0].size():
				nn[0] = []
				for i in input:
					nn[0].append([i, nn[-1][0].call(i)])
			else:
				return nn[-1]
		else:
			for neuron in nn[layer_num]:
				var v: float = 0 #Value
				for prev_n_num in range(nn[layer_num-1].size()):
					v += nn[layer_num-1][prev_n_num][1] * neuron[prev_n_num+2]
				v += neuron[-1]
				neuron[0] = v
				v = nn[-1][layer_num].call(v)
				neuron[1] = v
	return nn

func sigmoid(val: float) -> float:
	return 1/(1+exp(-val))

func relu(val: float) -> float:
	return max(0, val)

func nothing(val: float) -> float:
	return val

func f_prime(value: float, activation: Callable) -> float:
	match activation.get_method():
		"sigmoid":
			var activated: float = sigmoid(value)
			return activated * (1-activated)
		"relu":
			if value > 0.0:
				return 1.0
			return 0.0
		"nothing":
			return 1.0
	return 1.0

func educate(right_output: Array[float], input: Array[float], education_speed: float = 0.001, range_of_change: float = 0.01):
	for i in range(input.size()):
		layers[0][i][0] = input[i]
		layers[0][i][1] = layers[-1][0].call(input[i])
	layers = calculated_nn(input, layers)
	var errors: Array[Array]
	var activations: Array[Callable] = layers[-1]
	for layer in range(layers.size()-1):
		errors.append([])
	for layer_index in range(layers.size()-2, -1, -1):
		var layer = layers[layer_index]
		var errors_of_layer: PackedFloat32Array
		if layer_index == layers.size()-2:
			for neuron_num in range(0, layer.size()):
				var neuron: PackedFloat32Array = layer[neuron_num]
				errors_of_layer.append((neuron[1] - right_output[neuron_num])*f_prime(neuron[0], activations[layer_index]))
			errors[layer_index] = errors_of_layer
		elif layer_index != 0:
			for neuron_num in range(0, layer.size()):
				var neuron: PackedFloat32Array = layer[neuron_num]
				var a: float = 0
				for next_n_num in range(layers[layer_index+1].size()):
					a += errors[layer_index+1][next_n_num] * layers[layer_index+1][next_n_num][neuron_num+2]
				a *= f_prime(neuron[0], activations[layer_index])
				errors_of_layer.append(a)
			errors[layer_index] = errors_of_layer
	for layer_num in range(1, layers.size()-1):
		var this_layer_errors: PackedFloat32Array = errors[layer_num]
		var layer: Array[Array] = layers[layer_num]
		for neuron_num in range(layers[layer_num].size()):
			var neuron: PackedFloat32Array = layer[neuron_num]
			for weight_num in range(2, neuron.size()-1):
				layers[weight_num] -= clamp(education_speed * this_layer_errors[neuron_num] * layers[layer_num-1][weight_num-2][1], -range_of_change, range_of_change)
			layers[-1] -= clamp(education_speed * this_layer_errors[neuron_num], -range_of_change, range_of_change)
