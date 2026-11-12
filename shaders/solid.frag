// SPDX-License-Identifier: MIT
//
// A flat colour. Used for the background, for the areas no surface covers,
// and for anything the compositor draws itself.
//
// No colour conversion: a colour the compositor chose is already expressed in
// the output's space, so there is nothing to convert from.
//
// Compile with:
//   glslangValidator -V solid.frag -o solid.frag.spv
//   glslangValidator -V solid.frag -DCOLOR_LUT -o solid.lut.frag.spv

#version 450

layout(push_constant) uniform Push {
    vec4 pos_a;
    vec4 pos_b;
    vec4 tex_a;
    vec4 color;
    vec4 misc;
    vec4 csc0;
    vec4 csc1;
    vec4 csc2;
} push;

#ifdef COLOR_LUT
layout(set = 0, binding = 1) uniform sampler3D color_lut;
#endif

layout(location = 0) in vec2 in_uv;
layout(location = 0) out vec4 out_color;

void main() {
    // Premultiplied throughout: Wayland buffers are premultiplied, and mixing
    // conventions inside one pass is how edges end up with dark fringes.
    // Scaling all four components keeps it premultiplied.
    vec4 color = push.color * push.misc.x;
#ifdef COLOR_LUT
    // The cube is defined on unpremultiplied colour; dividing out and
    // multiplying back keeps this pass's premultiplied convention.
    if (color.a > 0.0) {
        vec3 rgb = texture(color_lut, clamp(color.rgb / color.a, vec3(0.0), vec3(1.0))).rgb;
        color = vec4(rgb * color.a, color.a);
    }
#endif
    out_color = color;
}
