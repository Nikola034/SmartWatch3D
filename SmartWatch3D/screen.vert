#version 330 core

layout(location = 0) in vec2 inPos;
layout(location = 1) in vec2 inTexCoord;

out vec2 texCoord;

uniform vec2 uPos;
uniform vec2 uScale;
uniform float uTexScaleX;
uniform float uTexOffsetX;

void main()
{
    vec2 scaledPos = inPos * uScale;
    gl_Position = vec4(scaledPos.x + uPos.x, scaledPos.y + uPos.y, 0.0, 1.0);
    texCoord = vec2(inTexCoord.x * uTexScaleX + uTexOffsetX, inTexCoord.y);
}
