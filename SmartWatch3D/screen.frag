#version 330 core

in vec2 texCoord;
out vec4 outColor;

uniform vec4 uColor;
uniform int uUseTexture;
uniform sampler2D uTexture;

void main()
{
    if (uUseTexture == 1) {
        vec4 texColor = texture(uTexture, texCoord);
        outColor = texColor * uColor;
    } else {
        outColor = uColor;
    }
}
