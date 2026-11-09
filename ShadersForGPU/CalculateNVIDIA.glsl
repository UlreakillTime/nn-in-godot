#[compute]
#version 450
// NVIDIA version (local_size_x = 32, AMD has local_size_x = 64)
layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

// Some values you'll need
layout(std430, binding = 0) readonly buffer First_layer {float size; float layer[];} FIRST;
layout(std430, binding = 1)          buffer Second_layer {float layer[];} SECOND;
layout(std430, binding = 2) readonly buffer Activation {float type;} ACTIVATION;

// All the code
void main() {
    // Total index (logicaly)
    int total_index = int((gl_WorkGroupID.x*64)+gl_LocalInvocationID);
    // If outside of needed range, just don't do anything
    if ((total_index) > (FIRST.size)) {
        return;
    }
    // Init the total value
    float val = 0.0;
    // Little optimisation is always helpful
    const int int_size = int(FIRST.size);
    // More optimisation
    const int size_and_index = (int_size+3)*total_index;
    //                        
    //                  0  1  2  3  4  5  6  7  8  9  10 11
    // Layer example : [v, v, w, w, w, b, v, v, w, w, w, b]
    for (int neuron_num=2; neuron_num<int_size+2; neuron_num++){
        // indexation             
        //  add to v ->           weight of this layer * val of prev layer
        val += SECOND.layer[neuron_num+size_and_index] * FIRST.layer[1+((neuron_num-2)*(int_size+3))];
    }
    // Add bias to v
    val += SECOND.layer[int_size+2+size_and_index];
    // Add unactivated windows (value) to OUTPUT
    SECOND.layer[size_and_index] = val;
    if (ACTIVATION.type == 0.0){
        // Add activated windows (value) to OUTPUT
        SECOND.layer[size_and_index+1] = val;
        // Leave
        return;
    }
    if (ACTIVATION.type == 1.0){
        // Add activated windows (value) to OUTPUT
        SECOND.layer[size_and_index+1] = max(0, val);
        // Leave
        return;
    }
    if (ACTIVATION.type == 2.0){
        // Add activated windows (value) to OUTPUT
        SECOND.layer[size_and_index+1] = 1/(1+exp(-val));
        // Leave
        return;
    }
    // Add unactivated windows (value) as activated to OUTPUT
    SECOND.layer[size_and_index+1] = val;
}