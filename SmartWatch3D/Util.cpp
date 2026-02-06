#include "Util.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/gtc/type_ptr.hpp>

#define _CRT_SECURE_NO_WARNINGS
#include <fstream>
#include <sstream>
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

unsigned int compileShader(GLenum type, const char* source)
{
    std::string content = "";
    std::ifstream file(source);
    std::stringstream ss;
    if (file.is_open())
    {
        ss << file.rdbuf();
        file.close();
        std::cout << "Successfully read file from path \"" << source << "\"!" << std::endl;
    }
    else {
        ss << "";
        std::cout << "Error reading file from path \"" << source << "\"!" << std::endl;
    }
    std::string temp = ss.str();
    const char* sourceCode = temp.c_str();

    int shader = glCreateShader(type);

    int success;
    char infoLog[512];
    glShaderSource(shader, 1, &sourceCode, NULL);
    glCompileShader(shader);

    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (success == GL_FALSE)
    {
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        if (type == GL_VERTEX_SHADER)
            printf("VERTEX");
        else if (type == GL_FRAGMENT_SHADER)
            printf("FRAGMENT");
        printf(" shader has error! Error: \n");
        printf("%s", infoLog);
    }
    return shader;
}

unsigned int createShader(const char* vsSource, const char* fsSource)
{
    unsigned int program;
    unsigned int vertexShader;
    unsigned int fragmentShader;

    program = glCreateProgram();

    vertexShader = compileShader(GL_VERTEX_SHADER, vsSource);
    fragmentShader = compileShader(GL_FRAGMENT_SHADER, fsSource);

    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);

    glLinkProgram(program);
    glValidateProgram(program);

    int success;
    char infoLog[512];
    glGetProgramiv(program, GL_VALIDATE_STATUS, &success);
    if (success == GL_FALSE)
    {
        glGetShaderInfoLog(program, 512, NULL, infoLog);
        std::cout << "Combined shader has error! Error: \n";
        std::cout << infoLog << std::endl;
    }

    glDetachShader(program, vertexShader);
    glDeleteShader(vertexShader);
    glDetachShader(program, fragmentShader);
    glDeleteShader(fragmentShader);

    return program;
}

unsigned int loadImageToTexture(const char* filePath)
{
    int textureWidth, textureHeight, textureChannels;
    unsigned char* textureData = stbi_load(filePath, &textureWidth, &textureHeight, &textureChannels, 0);

    if (textureData == NULL) {
        std::cout << "Error loading texture: " << filePath << std::endl;
        return 0;
    }

    unsigned int texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    GLenum format = GL_RGB;
    if (textureChannels == 1)
        format = GL_RED;
    else if (textureChannels == 3)
        format = GL_RGB;
    else if (textureChannels == 4)
        format = GL_RGBA;

    glTexImage2D(GL_TEXTURE_2D, 0, format, textureWidth, textureHeight, 0, format, GL_UNSIGNED_BYTE, textureData);
    glGenerateMipmap(GL_TEXTURE_2D);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    stbi_image_free(textureData);

    std::cout << "Successfully loaded texture: " << filePath << " (" << textureWidth << "x" << textureHeight << ", " << textureChannels << " channels)" << std::endl;

    return texture;
}

void setMat4(unsigned int shader, const std::string& name, const glm::mat4& mat) {
    glUniformMatrix4fv(glGetUniformLocation(shader, name.c_str()), 1, GL_FALSE, glm::value_ptr(mat));
}

void setVec3(unsigned int shader, const std::string& name, const glm::vec3& vec) {
    glUniform3fv(glGetUniformLocation(shader, name.c_str()), 1, glm::value_ptr(vec));
}

void setVec4(unsigned int shader, const std::string& name, const glm::vec4& vec) {
    glUniform4fv(glGetUniformLocation(shader, name.c_str()), 1, glm::value_ptr(vec));
}

void setFloat(unsigned int shader, const std::string& name, float value) {
    glUniform1f(glGetUniformLocation(shader, name.c_str()), value);
}

void setInt(unsigned int shader, const std::string& name, int value) {
    glUniform1i(glGetUniformLocation(shader, name.c_str()), value);
}
