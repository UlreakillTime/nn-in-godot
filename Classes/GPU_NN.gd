extends NN
class_name GPU_NN

## The Neural Network that uses GPU. Pretty basic, since if u want to make a project that u can build on python way easier, u should build it on python. Oh, btw, you should not change any variables by hand and expect everything to work

#TODO: 1) Add NVIDIA support 2) Add desirable_max_output

var layers: Array[PackedFloat32Array] ## All the layers (first is a list of all activations (the first layer doesn't get activated so there is 1 less))

var rd := RenderingServer.create_local_rendering_device() ## The thing that allows the code to use GPU
var calculate_shader := rd.shader_create_from_spirv( ## The shader used for calulating layers 1 by 1
	load("res://NNGLSL/Calculate.glsl").get_spirv()
)

enum activation {
	None = 0, ## Just leaves the value. output = val
	ReLU = 1, ## If val < 0 then 0, else just the val. output = max(0, val)
	Sigmoid = 2 ## If you don't know this activation, just google it, or place y=1/(1+e^-x) in desmos. output = 1/(1+exp(-val))
}

## Setup function. Use to initialize all the layers. Currently only supports dense layers
## Desirable output means that the output would be what you set it would if all activations were on 0 aka activation.none
## Returns OK if everything is fine, returns ERR_INVALID_DATA if: 1) amount of layers and amount of activations don't match 2) 1 or more of your activations aren't currently real (like Tanh of LeakyReLU) 3) 1 or more of your layers have a size of 0 or less
@warning_ignore("unused_parameter")
func setup(layers_sizes: PackedInt32Array, activations: Array[activation], desirable_max_output: float = 1.0) -> Error:
	if activations.size() != layers_sizes.size():
		return Error.ERR_INVALID_DATA
	#print(layers_sizes)
	layers.clear()
	layers.append([])
	layers.append([])
	if layers_sizes[0] <= 0:
		return Error.ERR_INVALID_DATA
	for i in range(layers_sizes[0]*3):
		layers[1].append(randf())
	for size_index in range(1, layers_sizes.size()):
		#print(size_index)
		#print(activation.values().has(activations[size_index]))
		#print(activation)
		#print(layers_sizes[size_index])
		if not activation.values().has(activations[size_index]) or layers_sizes[size_index] <= 0:
			return Error.ERR_INVALID_DATA
		#print(size_index)
		layers[0].append(float(activations[size_index]))
		layers.append([])
		for neuron in range((layers_sizes[size_index])*(layers_sizes[size_index-1]+3)):
			layers[size_index+1].append(randf())
	return Error.OK

func calculate_NN():
	print("Started calculate_NN()")
	@warning_ignore("integer_division")
	var layer_size: int = layers[1].size()/(3)
	for layer_index in range(2, layers.size()):
		@warning_ignore("narrowing_conversion")
		layers[layer_index] = calculate_layer(layers[layer_index-1], layers[layer_index], int(layers[0][layer_index-2]), layer_size)
		@warning_ignore("integer_division")
		layer_size = layers[layer_index].size()/(3+layer_size)
		print("Predicted layer_size: ", layer_size)

## The function which calculates layers values. values - first layer (used as a source of values), next_layer - the layer which will be calculated, activation - the kind of activation you wanna have
func calculate_layer(values: PackedFloat32Array, next_layer: PackedFloat32Array, layer_activation: activation, layer_size: int): 
	#var shader_file := load("res://FunkyGLSLShaders/timestwo.glsl") # Load a shader
	#var shader_spirv: RDShaderSPIRV = shader_file.get_spirv() # Load some shit from shader, thx to creators u don't need it later
	#print(shader_spirv.get_stage_compile_error(rd.SHADER_STAGE_COMPUTE))
	#var shader := rd.shader_create_from_spirv(shader_spirv)  # Create final shader from that shit
	#var shader_spirv = load("res://NNGLSL/Calculate.glsl").get_spirv()
	#print(shader_spirv.get_stage_compile_error(rd.SHADER_STAGE_COMPUTE))
	#calculate_shader = rd.shader_create_from_spirv(shader_spirv)
	
	var starting_time = Time.get_ticks_msec()
	
	# VALUES
	
	# Then we transform values into bytes
	# total_size = neurons*(3+prev_layer_total_size)
	# neurons = total_size/(3+prev_layer_total_size)
	values.insert(0, layer_size)
	var values_bytes := values.to_byte_array() # Transform into byte_array
	var values_buffer := rd.storage_buffer_create(values_bytes.size(), values_bytes) # Transform into raw food (?) (buffer)
	
	# After that we transform our raw food into a cooked food (uniform) for da shader
	var values_uniform := RDUniform.new() # Create a food
	values_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER # Say a type of food
	values_uniform.binding = 0 # Say a table where food needs to be
	values_uniform.add_id(values_buffer) # Create a connection between raw food and shader for some reason
	
	# LAYER
	
	# Then we transform input into bytes
	var layer_bytes := next_layer.to_byte_array() # Transform into byte_array
	var layer_buffer := rd.storage_buffer_create(layer_bytes.size(), layer_bytes) # Transform into raw food (?) (buffer)
	
	# After that we transform our raw food into a cooked food (uniform) for da shader
	var layer_uniform := RDUniform.new() # Create a food
	layer_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER # Say a type of food
	layer_uniform.binding = 1 # Say a table where food needs to be
	layer_uniform.add_id(layer_buffer) # Create a connection between raw food and shader for some reason
	
	# ACTIVATION
	
	# Then we transform input into bytes
	var activation_bytes := PackedFloat32Array([float(layer_activation)]).to_byte_array() # Transform into byte_array
	var activation_buffer := rd.storage_buffer_create(activation_bytes.size(), activation_bytes) # Transform into raw food (?) (buffer)
	
	# After that we transform our raw food into a cooked food (uniform) for da shader
	var activation_uniform := RDUniform.new() # Create a food
	activation_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER # Say a type of food
	activation_uniform.binding = 2 # Say a table where food needs to be
	activation_uniform.add_id(activation_buffer) # Create a connection between raw food and shader for some reason
	
	# OUTPUT
	
	# Create cooked food that'll be used as output
	var output_buffer := rd.storage_buffer_create(next_layer.size()*4) # Create buffer for smth idk how to explain it
	var output_uniform := RDUniform.new() # Create a food
	output_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER # Say a type of food
	output_uniform.binding = 3 # Say a table where food needs to be
	output_uniform.add_id(output_buffer) # Create a connection between raw food and shader for some reason
	
	var uniform_set := rd.uniform_set_create([values_uniform, layer_uniform, activation_uniform, output_uniform], calculate_shader, 0) # the last parameter (the 0) needs to match the "set" in our shader file
	
	# After that we make a road(?)
	var pipeline := rd.compute_pipeline_create(calculate_shader) # Create a road
	
	# And then we load that road into an array(?) of roads (compute_list)
	var compute_list := rd.compute_list_begin() # Create an array of roads
	rd.compute_list_bind_compute_pipeline(compute_list, pipeline) # Add a road to it
	rd.compute_list_bind_uniform_set(compute_list, uniform_set, 0) # Add food onto the road in index 0
	rd.compute_list_dispatch(compute_list, int(ceilf(layer_size/64.0)), 1, 1) # Send all that shit to GPU (input and output)
	
	# Now it's time to create a road from GPU
	#var output_pipeline := rd.compute_pipeline_create(shader) # Create a road
	
	# And then we load that road into an array of roads that we already have
	#var output_compute_list := rd.compute_list_begin() # Create an array of roads
	#rd.compute_list_bind_compute_pipeline(compute_list, output_pipeline) # Add a road to it
	#rd.compute_list_bind_uniform_set(compute_list, output_uniform_set, 0) # Add food onto the road in index 1
	
	
	# And end that shit
	rd.compute_list_end() # End da road arrray
	rd.submit() # git submit git push
	rd.sync()# git wait_complition
	
	# Running calculate shader
	var output_bytes := rd.buffer_get_data(layer_buffer)
	var output := output_bytes.to_float32_array()
	
	print("Time needed for calulating a single layer (in seconds): ", (Time.get_ticks_msec()-starting_time)/968.0)
	
	return output
