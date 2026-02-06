#pragma once
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <string>

// Shader utility functions
unsigned int compileShader(GLenum type, const char* source);
unsigned int createShader(const char* vsSource, const char* fsSource);

// Texture loading
unsigned int loadImageToTexture(const char* filePath);

// Shader uniform helpers
void setMat4(unsigned int shader, const std::string& name, const glm::mat4& mat);
void setVec3(unsigned int shader, const std::string& name, const glm::vec3& vec);
void setVec4(unsigned int shader, const std::string& name, const glm::vec4& vec);
void setFloat(unsigned int shader, const std::string& name, float value);
void setInt(unsigned int shader, const std::string& name, int value);
