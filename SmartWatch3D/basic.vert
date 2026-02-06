#version 330 core

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;

out vec3 fragPos;
out vec3 normal;
out vec2 texCoord;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProjection;

void main()
{
    fragPos = vec3(uModel * vec4(inPos, 1.0));
    normal = mat3(transpose(inverse(uModel))) * inNormal;
    texCoord = inTexCoord;
    gl_Position = uProjection * uView * vec4(fragPos, 1.0);
}
