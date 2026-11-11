#[compute]
#version 450

// AMD version (local_size_x = 64, NVIDIA has local_size_x = 32)
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

// Some values you'll need
layout(std430, binding = 0) readonly buffer First_layer {float activation; float size; float layer[];} VALUES_LAYER;
layout(std430, binding = 1)          buffer Second_layer {float activation; float size; float layer[];} WEIGHTS_LAYER;
layout(std430, binding = 2)          buffer Updates {float VALUES_lAYER;} UPDATES_FOR;

/*
PLAN:
1) Calculate gradient:
    1) For every neuron
2) Update weights for WEIGHTS_LAYER
*/

void main() {
    // Total index (logicaly)
    int total_index = int((gl_WorkGroupID.x*64)+gl_LocalInvocationID);
    // If outside of needed range, just don't do anything
    if ((total_index) > (FIRST.size)) {
        return;
    }
    // Init the total error
    float err = 0.0;
    // Little optimisation is always helpful
    const int int_size = int(FIRST.size);
    // More optimisation
    const int size_and_index = (int_size+3)*total_index;
    //                        
    //                  0  1  2  3  4  5  6  7  8  9  10 11
    // Layer example : [v, v, w, w, w, b, v, v, w, w, w, b]
    for (int neuron_num=2; neuron_num<int_size+2; neuron_num++){
        err += с
    }
}