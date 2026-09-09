// SPDX-License-Identifier: MIT
//
// The background blur behind an `ext-background-effect-v1` surface.
//
// Nine taps over a texture the capture pass has already downsampled by four,
// so the radius is sixteen source pixels and the cost does not grow with the
// output's resolution. The weights are the ones the GLES path uses, which is
// what makes the two renderers look the same on a machine that can run either.
//
// No colour conversion here: the captured texture is the framebuffer this
// shader is drawing back into, so it is already in the output's space.
//
// Compile with:
//   glslangValidator -V blur.frag -o blur.frag.spv

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

layout(set = 0, binding = 0) uniform sampler2D source;

layout(location = 0) in vec2 in_uv;
layout(location = 0) out vec4 out_color;

vec4 sample_at(vec2 offset) {
    vec2 edge = vec2(0.5) / vec2(textureSize(source, 0));
    return texture(source, clamp(in_uv + offset, edge, vec2(1.0) - edge));
}

void main() {
    vec2 radius = 4.0 / vec2(textureSize(source, 0));
    vec4 color = sample_at(vec2(0.0)) * 0.204164;
    color += sample_at(vec2(radius.x, 0.0)) * 0.123841;
    color += sample_at(vec2(-radius.x, 0.0)) * 0.123841;
    color += sample_at(vec2(0.0, radius.y)) * 0.123841;
    color += sample_at(vec2(0.0, -radius.y)) * 0.123841;
    color += sample_at(vec2(radius.x, radius.y)) * 0.075114;
    color += sample_at(vec2(-radius.x, radius.y)) * 0.075114;
    color += sample_at(vec2(radius.x, -radius.y)) * 0.075114;
    color += sample_at(vec2(-radius.x, -radius.y)) * 0.075114;

    // `misc.x` is alpha, as it is for the textured pipeline.
    out_color = color * push.misc.x;
}
