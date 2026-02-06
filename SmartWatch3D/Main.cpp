/*
 * ============================================================================
 * SmartWatch 3D Simulator
 * ============================================================================
 * Author: Nikola Bandulaja SV74/2022
 *
 * This application renders a 3D smartwatch simulation with:
 * - Perspective projection and Phong lighting model
 * - Two-pass rendering: watch UI to FBO, then 3D scene to screen
 * - Two light sources: sun (main) and watch screen (emissive)
 * - Infinite running simulation with scrolling ground and buildings
 * - Camera controls and watch interaction modes
 *
 * ARCHITECTURE OVERVIEW:
 * ----------------------
 * 1. First Pass (FBO): Render 2D watch UI elements to off-screen texture
 *    - Uses screenShader (simple 2D shader)
 *    - Renders clock digits, EKG graph, battery indicator, navigation arrows
 *
 * 2. Second Pass (Screen): Render 3D scene with watch texture
 *    - Uses basicShader (Phong lighting shader)
 *    - Renders ground, road, buildings, hand, watch frame, watch screen
 *    - Watch screen uses FBO texture and is marked as emissive
 *
 * CONTROLS:
 * ---------
 * - SPACE: Toggle watch view mode (brings watch in front of camera)
 * - D: Hold to simulate running (only works on heart rate screen)
 * - Mouse: Look up/down (pitch) when not in watch view mode
 * - Click: Navigate watch screens (only in watch view mode)
 * - F1: Toggle depth testing
 * - F2: Toggle face culling
 * - ESC: Exit application
 * ============================================================================
 */

#define _CRT_SECURE_NO_WARNINGS

// ==================== INCLUDES ====================
// OpenGL Extension Wrangler - must be included before GLFW
#include <GL/glew.h>
// GLFW - window management and input handling
#include <GLFW/glfw3.h>

// GLM - OpenGL Mathematics library for matrix/vector operations
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>  // For glm::translate, rotate, scale, perspective, lookAt
#include <glm/gtc/type_ptr.hpp>          // For glm::value_ptr (convert to raw pointer)

#define _USE_MATH_DEFINES
#include <cmath>
#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cstdio>
#include <chrono>
#include <thread>
#include <vector>
#include <cstring>

#include "Util.h"  // Shader compilation and texture loading utilities

// ==================== CONSTANTS ====================

// Frame rate limiter - limits to 75 FPS for smooth animation
const double TARGET_FPS = 75.0;
const double TARGET_FRAME_TIME = 1.0 / TARGET_FPS;  // ~13.3ms per frame

// Ground/road configuration for infinite scrolling effect
const float GROUND_SEGMENT_LENGTH = 20.0f;  // Length of one ground segment
const int NUM_GROUND_SEGMENTS = 5;          // Number of segments to tile
const float ROAD_WIDTH = 8.0f;              // Width of the road

// Building configuration
const int NUM_BUILDINGS_PER_SIDE = 6;       // Buildings on each side of road
const float BUILDING_SPACING = 15.0f;       // Distance between buildings

// ==================== GLOBAL VARIABLES ====================

// Window dimensions (fullscreen)
int screenWidth = 1920;
int screenHeight = 1080;

// ----- Watch Screen State (ported from 2D version) -----
int currentScreen = 0;  // 0 = clock, 1 = heart rate, 2 = battery

// Time tracking for clock display
time_t currentTimeVal = time(NULL);
struct tm* localTimeVal = localtime(&currentTimeVal);
int hours = localTimeVal->tm_hour;
int minutes = localTimeVal->tm_min;
int seconds = localTimeVal->tm_sec;
double lastSecondTime = 0;  // For updating clock every second

// Heart rate simulation
float bpm = 70.0f;           // Current displayed BPM
float targetBpm = 70.0f;     // Target BPM (increases when running)
float ekgOffset = 0.0f;      // Horizontal scroll of EKG graph
float ekgScale = 1.0f;       // Scale factor for EKG animation speed
bool isRunning = false;      // Whether D key is held

// Battery simulation
int batteryPercent = 100;    // Current battery level
double lastBatteryDrain = 0; // Timer for battery drain

// ----- Camera State -----
glm::vec3 cameraPos = glm::vec3(0.0f, 1.6f, 0.0f);  // Eye level height (1.6m)
float cameraYaw = -90.0f;      // Horizontal rotation (looking down -Z axis)
float cameraPitch = 0.0f;      // Vertical rotation (combined base + bob)
float cameraBasePitch = 0.0f;  // Base pitch from mouse input
bool watchViewMode = false;    // When true, watch is in front of camera

// ----- Mouse State -----
double mouseX = 0, mouseY = 0;       // Current mouse position
double lastMouseX = 0, lastMouseY = 0;
bool firstMouse = true;              // For initializing mouse delta
bool mouseClicked = false;           // Left click flag for UI interaction

// ----- Running Animation -----
float runTime = 0.0f;          // Accumulated time while running
float groundOffset = 0.0f;     // How far ground has scrolled (for infinite effect)
float cameraBobOffset = 0.0f;  // Vertical camera bob while running

// ----- Render Settings (toggleable) -----
bool depthTestEnabled = true;    // F1 toggles depth testing
bool faceCullingEnabled = true;  // F2 toggles back-face culling

// ----- OpenGL Textures -----
unsigned int groundTexture;       // Grass texture for ground
unsigned int roadTexture;         // Asphalt texture for road
unsigned int ekgTexture;          // EKG waveform pattern
unsigned int arrowRightTexture;   // Navigation arrow (right)
unsigned int arrowLeftTexture;    // Navigation arrow (left)
unsigned int heartCursorTexture;  // Heart icon for BPM display
unsigned int studentInfoTexture;  // Student name overlay
unsigned int buildingTexture;     // Generic building texture
unsigned int watchFrameTexture;   // Watch bezel texture

// ----- Shader Programs -----
unsigned int basicShader;   // 3D Phong lighting shader
unsigned int screenShader;  // 2D shader for watch UI rendering

// ----- Vertex Array Objects -----
unsigned int VAOground;      // Ground plane (large quad)
unsigned int VAOcube;        // Unit cube (for buildings, hand, watch frame)
unsigned int VAOwatchQuad;   // 3D quad for watch screen in world space
unsigned int VAOscreenQuad;  // 2D quad for FBO rendering
unsigned int VAOhand;        // Hand mesh (reuses cube VAO)

// ----- Framebuffer Object for Watch Screen -----
// The watch UI is first rendered to this FBO, then the resulting texture
// is applied to the 3D watch quad in the scene
unsigned int watchFBO;           // Framebuffer object handle
unsigned int watchScreenTexture; // Color attachment (render target)
const int WATCH_SCREEN_SIZE = 512;  // Resolution of watch screen texture

// ----- Building Data -----
// Stores procedurally generated building properties
struct Building {
    glm::vec3 position;  // World position
    glm::vec3 scale;     // Size (width, height, depth)
    glm::vec3 color;     // RGB color
};
std::vector<Building> buildings;  // All buildings in the scene

// ==================== HELPER FUNCTIONS ====================

/**
 * Clean up and exit the program with an error message
 */
int endProgram(std::string message) {
    std::cout << message << std::endl;
    glfwTerminate();
    return -1;
}

// ==================== PROCEDURAL TEXTURE CREATION ====================
/*
 * These functions create textures procedurally (without loading image files).
 * This ensures the application is self-contained and doesn't require external assets.
 * Each texture is generated pixel-by-pixel and uploaded to the GPU.
 */

/**
 * Creates the EKG (electrocardiogram) waveform texture
 * The waveform shows the characteristic PQRST pattern of a heartbeat:
 * - P wave: small bump (atrial depolarization)
 * - QRS complex: large spike (ventricular depolarization)
 * - T wave: medium bump (ventricular repolarization)
 *
 * This texture is tiled horizontally to create a scrolling EKG display
 */
unsigned int createEKGTexture() {
    const int width = 256;
    const int height = 128;
    unsigned char* data = new unsigned char[width * height * 4];

    for (int i = 0; i < width * height * 4; i += 4) {
        data[i] = 0;
        data[i + 1] = 0;
        data[i + 2] = 0;
        data[i + 3] = 0;
    }

    auto setPixel = [&](int x, int y, unsigned char r, unsigned char g, unsigned char b) {
        if (x >= 0 && x < width && y >= 0 && y < height) {
            int idx = (y * width + x) * 4;
            data[idx] = r;
            data[idx + 1] = g;
            data[idx + 2] = b;
            data[idx + 3] = 255;
        }
    };

    auto drawThickLine = [&](int x, int y1, int y2) {
        int minY = (std::min)(y1, y2);
        int maxY = (std::max)(y1, y2);
        for (int y = minY; y <= maxY; y++) {
            for (int dx = -2; dx <= 2; dx++) {
                setPixel(x + dx, y, 0, 255, 0);
            }
        }
    };

    int baseline = height / 2;
    int lastY = baseline;

    for (int x = 0; x < width; x++) {
        int y = baseline;
        float t = (float)x / width;

        if (t < 0.1f) {
            y = baseline;
        }
        else if (t < 0.15f) {
            float local = (t - 0.1f) / 0.05f;
            y = baseline - (int)(10 * sin(local * M_PI));
        }
        else if (t < 0.25f) {
            y = baseline;
        }
        else if (t < 0.30f) {
            float local = (t - 0.25f) / 0.05f;
            y = baseline + (int)(8 * sin(local * M_PI));
        }
        else if (t < 0.40f) {
            float local = (t - 0.30f) / 0.10f;
            if (local < 0.5f) {
                y = baseline - (int)(50 * (local * 2));
            }
            else {
                y = baseline - (int)(50 * (1.0f - (local - 0.5f) * 2));
            }
        }
        else if (t < 0.48f) {
            float local = (t - 0.40f) / 0.08f;
            y = baseline + (int)(15 * sin(local * M_PI));
        }
        else if (t < 0.65f) {
            float local = (t - 0.48f) / 0.17f;
            y = baseline - (int)(15 * sin(local * M_PI));
        }
        else {
            y = baseline;
        }

        drawThickLine(x, lastY, y);
        lastY = y;
    }

    unsigned int texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    delete[] data;
    return texture;
}

unsigned int createArrowTexture(bool pointRight) {
    const int size = 64;
    unsigned char* data = new unsigned char[size * size * 4];

    for (int i = 0; i < size * size * 4; i += 4) {
        data[i] = 0;
        data[i + 1] = 0;
        data[i + 2] = 0;
        data[i + 3] = 0;
    }

    auto setPixel = [&](int x, int y) {
        if (x >= 0 && x < size && y >= 0 && y < size) {
            int idx = (y * size + x) * 4;
            data[idx] = 255;
            data[idx + 1] = 255;
            data[idx + 2] = 255;
            data[idx + 3] = 255;
        }
    };

    int cy = size / 2;

    for (int thickness = -3; thickness <= 3; thickness++) {
        for (int x = 15; x < 50; x++) {
            setPixel(x, cy + thickness);
        }
        for (int i = 0; i < 15; i++) {
            if (pointRight) {
                setPixel(49 - i, cy - i + thickness);
                setPixel(49 - i, cy + i + thickness);
            }
            else {
                setPixel(15 + i, cy - i + thickness);
                setPixel(15 + i, cy + i + thickness);
            }
        }
    }

    unsigned int texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, size, size, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    delete[] data;
    return texture;
}

unsigned int createHeartTexture() {
    const int size = 32;
    unsigned char* data = new unsigned char[size * size * 4];

    for (int i = 0; i < size * size * 4; i += 4) {
        data[i] = 0;
        data[i + 1] = 0;
        data[i + 2] = 0;
        data[i + 3] = 0;
    }

    int cx = size / 2;
    int cy = size / 2;

    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            float fx = (x - cx) / (float)(size / 2);
            float fy = (y - cy) / (float)(size / 2);

            float val = pow(fx * fx + fy * fy - 0.5f, 3) - fx * fx * fy * fy * fy;

            if (val < 0) {
                int idx = (y * size + x) * 4;
                data[idx] = 255;
                data[idx + 1] = 50;
                data[idx + 2] = 80;
                data[idx + 3] = 255;
            }
        }
    }

    unsigned int texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, size, size, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    delete[] data;
    return texture;
}

const unsigned char FONT_DATA[128][7] = {
    {0,0,0,0,0,0,0}, {0,0,0,0,0,0,0}, {0,0,0,0,0,0,0}, {0,0,0,0,0,0,0}, // 0-3
    {0,0,0,0,0,0,0}, {0,0,0,0,0,0,0}, {0,0,0,0,0,0,0}, {0,0,0,0,0,0,0}, // 4-7
    {0,0,0,0,0,0,0}, {0,0,0,0,0,0,0}, {0,0,0,0,0,0,0}, {0,0,0,0,0,0,0}, // 8-11
    {0,0,0,0,0,0,0}, {0,0,0,0,0,0,0}, {0,0,0,0,0,0,0}, {0,0,0,0,0,0,0}, // 12-15
    {0,0,0,0,0,0,0}, {0,0,0,0,0,0,0}, {0,0,0,0,0,0,0}, {0,0,0,0,0,0,0}, // 16-19
    {0,0,0,0,0,0,0}, {0,0,0,0,0,0,0}, {0,0,0,0,0,0,0}, {0,0,0,0,0,0,0}, // 20-23
    {0,0,0,0,0,0,0}, {0,0,0,0,0,0,0}, {0,0,0,0,0,0,0}, {0,0,0,0,0,0,0}, // 24-27
    {0,0,0,0,0,0,0}, {0,0,0,0,0,0,0}, {0,0,0,0,0,0,0}, {0,0,0,0,0,0,0}, // 28-31
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00}, // 32 space
    {0,0,0,0,0,0,0}, {0,0,0,0,0,0,0}, {0,0,0,0,0,0,0}, {0,0,0,0,0,0,0}, // 33-36
    {0,0,0,0,0,0,0}, {0,0,0,0,0,0,0}, {0,0,0,0,0,0,0}, {0,0,0,0,0,0,0}, // 37-40
    {0,0,0,0,0,0,0}, {0,0,0,0,0,0,0}, {0,0,0,0,0,0,0}, {0,0,0,0,0,0,0}, // 41-44
    {0,0,0,0,0,0,0}, {0,0,0,0,0,0,0}, // 45-46
    {0x01,0x01,0x02,0x04,0x08,0x10,0x10}, // 47 /
    {0x0E,0x11,0x13,0x15,0x19,0x11,0x0E}, // 48 0
    {0x04,0x0C,0x04,0x04,0x04,0x04,0x0E}, // 49 1
    {0x0E,0x11,0x01,0x02,0x04,0x08,0x1F}, // 50 2
    {0x1F,0x02,0x04,0x02,0x01,0x11,0x0E}, // 51 3
    {0x02,0x06,0x0A,0x12,0x1F,0x02,0x02}, // 52 4
    {0x1F,0x10,0x1E,0x01,0x01,0x11,0x0E}, // 53 5
    {0x06,0x08,0x10,0x1E,0x11,0x11,0x0E}, // 54 6
    {0x1F,0x01,0x02,0x04,0x08,0x08,0x08}, // 55 7
    {0x0E,0x11,0x11,0x0E,0x11,0x11,0x0E}, // 56 8
    {0x0E,0x11,0x11,0x0F,0x01,0x02,0x0C}, // 57 9
    {0,0,0,0,0,0,0}, {0,0,0,0,0,0,0}, {0,0,0,0,0,0,0}, {0,0,0,0,0,0,0}, // 58-61
    {0,0,0,0,0,0,0}, {0,0,0,0,0,0,0}, {0,0,0,0,0,0,0}, // 62-64
    {0x0E,0x11,0x11,0x1F,0x11,0x11,0x11}, // 65 A
    {0x1E,0x11,0x11,0x1E,0x11,0x11,0x1E}, // 66 B
    {0x0E,0x11,0x10,0x10,0x10,0x11,0x0E}, // 67 C
    {0x1C,0x12,0x11,0x11,0x11,0x12,0x1C}, // 68 D
    {0x1F,0x10,0x10,0x1E,0x10,0x10,0x1F}, // 69 E
    {0x1F,0x10,0x10,0x1E,0x10,0x10,0x10}, // 70 F
    {0x0E,0x11,0x10,0x17,0x11,0x11,0x0F}, // 71 G
    {0x11,0x11,0x11,0x1F,0x11,0x11,0x11}, // 72 H
    {0x0E,0x04,0x04,0x04,0x04,0x04,0x0E}, // 73 I
    {0x07,0x02,0x02,0x02,0x02,0x12,0x0C}, // 74 J
    {0x11,0x12,0x14,0x18,0x14,0x12,0x11}, // 75 K
    {0x10,0x10,0x10,0x10,0x10,0x10,0x1F}, // 76 L
    {0x11,0x1B,0x15,0x15,0x11,0x11,0x11}, // 77 M
    {0x11,0x11,0x19,0x15,0x13,0x11,0x11}, // 78 N
    {0x0E,0x11,0x11,0x11,0x11,0x11,0x0E}, // 79 O
    {0x1E,0x11,0x11,0x1E,0x10,0x10,0x10}, // 80 P
    {0x0E,0x11,0x11,0x11,0x15,0x12,0x0D}, // 81 Q
    {0x1E,0x11,0x11,0x1E,0x14,0x12,0x11}, // 82 R
    {0x0F,0x10,0x10,0x0E,0x01,0x01,0x1E}, // 83 S
    {0x1F,0x04,0x04,0x04,0x04,0x04,0x04}, // 84 T
    {0x11,0x11,0x11,0x11,0x11,0x11,0x0E}, // 85 U
    {0x11,0x11,0x11,0x11,0x11,0x0A,0x04}, // 86 V
    {0x11,0x11,0x11,0x15,0x15,0x15,0x0A}, // 87 W
    {0x11,0x11,0x0A,0x04,0x0A,0x11,0x11}, // 88 X
    {0x11,0x11,0x11,0x0A,0x04,0x04,0x04}, // 89 Y
    {0x1F,0x01,0x02,0x04,0x08,0x10,0x1F}, // 90 Z
    {0,0,0,0,0,0,0}, {0,0,0,0,0,0,0}, {0,0,0,0,0,0,0}, {0,0,0,0,0,0,0}, // 91-94
    {0,0,0,0,0,0,0}, {0,0,0,0,0,0,0}, // 95-96
    {0x00,0x00,0x0E,0x01,0x0F,0x11,0x0F}, // 97 a
    {0x10,0x10,0x16,0x19,0x11,0x11,0x1E}, // 98 b
    {0x00,0x00,0x0E,0x10,0x10,0x11,0x0E}, // 99 c
    {0x01,0x01,0x0D,0x13,0x11,0x11,0x0F}, // 100 d
    {0x00,0x00,0x0E,0x11,0x1F,0x10,0x0E}, // 101 e
    {0x06,0x09,0x08,0x1C,0x08,0x08,0x08}, // 102 f
    {0x00,0x00,0x0F,0x11,0x0F,0x01,0x0E}, // 103 g
    {0x10,0x10,0x16,0x19,0x11,0x11,0x11}, // 104 h
    {0x04,0x00,0x0C,0x04,0x04,0x04,0x0E}, // 105 i
    {0x02,0x00,0x06,0x02,0x02,0x12,0x0C}, // 106 j
    {0x10,0x10,0x12,0x14,0x18,0x14,0x12}, // 107 k
    {0x0C,0x04,0x04,0x04,0x04,0x04,0x0E}, // 108 l
    {0x00,0x00,0x1A,0x15,0x15,0x11,0x11}, // 109 m
    {0x00,0x00,0x16,0x19,0x11,0x11,0x11}, // 110 n
    {0x00,0x00,0x0E,0x11,0x11,0x11,0x0E}, // 111 o
    {0x00,0x00,0x1E,0x11,0x1E,0x10,0x10}, // 112 p
    {0x00,0x00,0x0D,0x13,0x0F,0x01,0x01}, // 113 q
    {0x00,0x00,0x16,0x19,0x10,0x10,0x10}, // 114 r
    {0x00,0x00,0x0E,0x10,0x0E,0x01,0x1E}, // 115 s
    {0x08,0x08,0x1C,0x08,0x08,0x09,0x06}, // 116 t
    {0x00,0x00,0x11,0x11,0x11,0x13,0x0D}, // 117 u
    {0x00,0x00,0x11,0x11,0x11,0x0A,0x04}, // 118 v
    {0x00,0x00,0x11,0x11,0x15,0x15,0x0A}, // 119 w
    {0x00,0x00,0x11,0x0A,0x04,0x0A,0x11}, // 120 x
    {0x00,0x00,0x11,0x11,0x0F,0x01,0x0E}, // 121 y
    {0x00,0x00,0x1F,0x02,0x04,0x08,0x1F}, // 122 z
    {0,0,0,0,0,0,0}, {0,0,0,0,0,0,0}, {0,0,0,0,0,0,0}, {0,0,0,0,0,0,0}, // 123-126
    {0,0,0,0,0,0,0}  // 127
};

unsigned int createStudentInfoTexture() {
    const int width = 256;
    const int height = 64;
    unsigned char* data = new unsigned char[width * height * 4];

    for (int i = 0; i < width * height * 4; i += 4) {
        data[i] = 30;
        data[i + 1] = 30;
        data[i + 2] = 50;
        data[i + 3] = 180;
    }

    auto setPixel = [&](int x, int y, unsigned char r, unsigned char g, unsigned char b) {
        if (x >= 0 && x < width && y >= 0 && y < height) {
            int idx = (y * width + x) * 4;
            data[idx] = r;
            data[idx + 1] = g;
            data[idx + 2] = b;
            data[idx + 3] = 255;
        }
    };

    auto drawChar = [&](char c, int startX, int startY, int scale,
        unsigned char r, unsigned char g, unsigned char b) -> int {
            unsigned char idx = (unsigned char)c;
            if (idx >= 128) return 6 * scale;

            for (int row = 0; row < 7; row++) {
                unsigned char rowData = FONT_DATA[idx][row];
                for (int col = 0; col < 5; col++) {
                    if (rowData & (0x10 >> col)) {
                        for (int sy = 0; sy < scale; sy++) {
                            for (int sx = 0; sx < scale; sx++) {
                                setPixel(startX + col * scale + sx,
                                    startY - row * scale - sy, r, g, b);
                            }
                        }
                    }
                }
            }
            return 6 * scale;
        };

    auto drawString = [&](const char* str, int startX, int startY, int scale,
        unsigned char r, unsigned char g, unsigned char b) {
            int x = startX;
            while (*str) {
                x += drawChar(*str, x, startY, scale, r, g, b);
                str++;
            }
        };

    int scale = 2;
    drawString("Nikola Bandulaja", 10, 50, scale, 255, 255, 255);
    drawString("SV74/2022", 55, 22, scale, 200, 200, 220);

    unsigned int texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    delete[] data;
    return texture;
}

unsigned int createDigitTexture(const char* digitStr) {
    const int charWidth = 30;
    const int charHeight = 50;
    int len = (int)strlen(digitStr);
    int width = charWidth * len;
    int height = charHeight;

    unsigned char* data = new unsigned char[width * height * 4];

    for (int i = 0; i < width * height * 4; i += 4) {
        data[i] = 0;
        data[i + 1] = 0;
        data[i + 2] = 0;
        data[i + 3] = 0;
    }

    auto setPixel = [&](int x, int y, unsigned char r, unsigned char g, unsigned char b) {
        if (x >= 0 && x < width && y >= 0 && y < height) {
            int idx = (y * width + x) * 4;
            data[idx] = r;
            data[idx + 1] = g;
            data[idx + 2] = b;
            data[idx + 3] = 255;
        }
    };

    const bool segments[12][7] = {
        {1,1,1,0,1,1,1},
        {0,0,1,0,0,1,0},
        {1,0,1,1,1,0,1},
        {1,0,1,1,0,1,1},
        {0,1,1,1,0,1,0},
        {1,1,0,1,0,1,1},
        {1,1,0,1,1,1,1},
        {1,0,1,0,0,1,0},
        {1,1,1,1,1,1,1},
        {1,1,1,1,0,1,1},
        {0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0},
    };

    auto drawSegment = [&](int offsetX, int seg) {
        int thick = 4;
        int margin = 3;
        int segW = charWidth - 2 * margin;

        switch (seg) {
        case 0:
            for (int t = 0; t < thick; t++)
                for (int x = margin; x < margin + segW; x++)
                    setPixel(offsetX + x, height - margin - t, 200, 230, 255);
            break;
        case 1:
            for (int t = 0; t < thick; t++)
                for (int y = height / 2 + margin / 2; y < height - margin; y++)
                    setPixel(offsetX + margin + t, y, 200, 230, 255);
            break;
        case 2:
            for (int t = 0; t < thick; t++)
                for (int y = height / 2 + margin / 2; y < height - margin; y++)
                    setPixel(offsetX + charWidth - margin - t, y, 200, 230, 255);
            break;
        case 3:
            for (int t = 0; t < thick; t++)
                for (int x = margin; x < margin + segW; x++)
                    setPixel(offsetX + x, height / 2 + t - thick / 2, 200, 230, 255);
            break;
        case 4:
            for (int t = 0; t < thick; t++)
                for (int y = margin; y < height / 2 - margin / 2; y++)
                    setPixel(offsetX + margin + t, y, 200, 230, 255);
            break;
        case 5:
            for (int t = 0; t < thick; t++)
                for (int y = margin; y < height / 2 - margin / 2; y++)
                    setPixel(offsetX + charWidth - margin - t, y, 200, 230, 255);
            break;
        case 6:
            for (int t = 0; t < thick; t++)
                for (int x = margin; x < margin + segW; x++)
                    setPixel(offsetX + x, margin + t, 200, 230, 255);
            break;
        }
    };

    auto drawColon = [&](int offsetX) {
        int dotSize = 4;
        int cx = offsetX + charWidth / 2;
        for (int dy = -dotSize / 2; dy <= dotSize / 2; dy++) {
            for (int dx = -dotSize / 2; dx <= dotSize / 2; dx++) {
                setPixel(cx + dx, height * 3 / 4 + dy, 200, 230, 255);
                setPixel(cx + dx, height * 1 / 4 + dy, 200, 230, 255);
            }
        }
    };

    for (int i = 0; i < len; i++) {
        char c = digitStr[i];
        int offsetX = i * charWidth;

        if (c == ':') {
            drawColon(offsetX);
        }
        else if (c >= '0' && c <= '9') {
            int digit = c - '0';
            for (int s = 0; s < 7; s++) {
                if (segments[digit][s]) {
                    drawSegment(offsetX, s);
                }
            }
        }
    }

    unsigned int texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    delete[] data;
    return texture;
}

unsigned int createGroundTexture() {
    const int size = 256;
    unsigned char* data = new unsigned char[size * size * 3];

    srand(12345);
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            int idx = (y * size + x) * 3;
            int base = 60 + rand() % 30;
            data[idx] = base;
            data[idx + 1] = base + 20 + rand() % 20;
            data[idx + 2] = base - 20;
        }
    }

    unsigned int texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, size, size, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
    glGenerateMipmap(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    delete[] data;
    return texture;
}

unsigned int createRoadTexture() {
    const int width = 256;
    const int height = 256;
    unsigned char* data = new unsigned char[width * height * 3];

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * 3;
            int base = 50 + rand() % 15;
            data[idx] = base;
            data[idx + 1] = base;
            data[idx + 2] = base;
        }
    }

    // Center line
    for (int y = 0; y < height; y++) {
        for (int x = width / 2 - 4; x < width / 2 + 4; x++) {
            if ((y / 32) % 2 == 0) {
                int idx = (y * width + x) * 3;
                data[idx] = 255;
                data[idx + 1] = 255;
                data[idx + 2] = 200;
            }
        }
    }

    unsigned int texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
    glGenerateMipmap(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    delete[] data;
    return texture;
}

unsigned int createBuildingTexture() {
    const int size = 128;
    unsigned char* data = new unsigned char[size * size * 3];

    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            int idx = (y * size + x) * 3;
            data[idx] = 120;
            data[idx + 1] = 110;
            data[idx + 2] = 100;
        }
    }

    // Windows
    for (int wy = 0; wy < 4; wy++) {
        for (int wx = 0; wx < 4; wx++) {
            int startX = 8 + wx * 30;
            int startY = 8 + wy * 30;
            for (int dy = 0; dy < 20; dy++) {
                for (int dx = 0; dx < 18; dx++) {
                    int idx = ((startY + dy) * size + (startX + dx)) * 3;
                    if (idx < size * size * 3) {
                        data[idx] = 180;
                        data[idx + 1] = 200;
                        data[idx + 2] = 220;
                    }
                }
            }
        }
    }

    unsigned int texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, size, size, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
    glGenerateMipmap(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    delete[] data;
    return texture;
}

// ==================== VAO CREATION FUNCTIONS ====================
/*
 * VAO (Vertex Array Object) stores the configuration of vertex attributes.
 * Each VAO remembers:
 * - Which VBO (Vertex Buffer Object) contains the vertex data
 * - How the vertex data is formatted (position, normal, texcoord)
 * - Which attributes are enabled
 *
 * Vertex format used: [Position(3) | Normal(3) | TexCoord(2)] = 8 floats per vertex
 * - Position: 3D coordinates in model space
 * - Normal: Surface normal for lighting calculations
 * - TexCoord: UV coordinates for texture mapping
 */

/**
 * Creates the ground plane VAO
 * A large quad (100m x 20m) that tiles to create infinite ground
 * Normal points up (0, 1, 0) for correct lighting
 * Texture coordinates are scaled to tile the grass texture
 */
void createGroundVAO() {
    float halfW = 50.0f;  // Half width = 50m, total width = 100m
    float len = GROUND_SEGMENT_LENGTH;  // Length of one segment

    float vertices[] = {
        // Position              Normal           TexCoord
        -halfW, 0.0f,  0.0f,    0.0f, 1.0f, 0.0f,  0.0f, 0.0f,
         halfW, 0.0f,  0.0f,    0.0f, 1.0f, 0.0f,  10.0f, 0.0f,
         halfW, 0.0f, -len,     0.0f, 1.0f, 0.0f,  10.0f, 4.0f,
        -halfW, 0.0f, -len,     0.0f, 1.0f, 0.0f,  0.0f, 4.0f,
    };

    unsigned int indices[] = { 0, 1, 2, 0, 2, 3 };

    unsigned int VBO, EBO;
    glGenVertexArrays(1, &VAOground);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAOground);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);

    glBindVertexArray(0);
}

/**
 * Creates a unit cube VAO (1x1x1) centered at origin
 * Used for: buildings, hand, watch frame
 *
 * Each face has 6 vertices (2 triangles) with proper normals for lighting.
 * Face culling requires correct winding order (CCW when viewed from outside).
 *
 * IMPORTANT: The winding order must be counter-clockwise (CCW) when viewed
 * from outside the cube for back-face culling to work correctly.
 */
void createCubeVAO() {
    float vertices[] = {
        // Back face (facing -Z direction)
        -0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  0.0f, 0.0f,
         0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  1.0f, 0.0f,
         0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  1.0f, 1.0f,
         0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  1.0f, 1.0f,
        -0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  0.0f, 1.0f,
        -0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  0.0f, 0.0f,
        // Front face
        -0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  0.0f, 0.0f,
         0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  1.0f, 0.0f,
         0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  1.0f, 1.0f,
         0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  1.0f, 1.0f,
        -0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  0.0f, 1.0f,
        -0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  0.0f, 0.0f,
        // Left face (viewed from -X, CCW winding)
        -0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,  0.0f, 0.0f,
        -0.5f, -0.5f,  0.5f, -1.0f,  0.0f,  0.0f,  1.0f, 0.0f,
        -0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,  1.0f, 1.0f,
        -0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,  1.0f, 1.0f,
        -0.5f,  0.5f, -0.5f, -1.0f,  0.0f,  0.0f,  0.0f, 1.0f,
        -0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,  0.0f, 0.0f,
        // Right face (viewed from +X, CCW winding)
         0.5f, -0.5f,  0.5f,  1.0f,  0.0f,  0.0f,  0.0f, 0.0f,
         0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f,
         0.5f,  0.5f, -0.5f,  1.0f,  0.0f,  0.0f,  1.0f, 1.0f,
         0.5f,  0.5f, -0.5f,  1.0f,  0.0f,  0.0f,  1.0f, 1.0f,
         0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,  0.0f, 1.0f,
         0.5f, -0.5f,  0.5f,  1.0f,  0.0f,  0.0f,  0.0f, 0.0f,
         // Bottom face
         -0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,  0.0f, 1.0f,
          0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,  1.0f, 1.0f,
          0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,  1.0f, 0.0f,
          0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,  1.0f, 0.0f,
         -0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,  0.0f, 0.0f,
         -0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,  0.0f, 1.0f,
         // Top face
         -0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,  0.0f, 1.0f,
          0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,  1.0f, 1.0f,
          0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,  1.0f, 0.0f,
          0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,  1.0f, 0.0f,
         -0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,  0.0f, 0.0f,
         -0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,  0.0f, 1.0f
    };

    unsigned int VBO;
    glGenVertexArrays(1, &VAOcube);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAOcube);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);

    glBindVertexArray(0);
}

/**
 * Creates a 3D quad for the watch screen display
 * This quad exists in world space and will have the FBO texture mapped to it.
 * Normal points forward (+Z) for emissive lighting calculation.
 * Size: 0.3m x 0.3m (30cm square watch face)
 */
void createWatchQuadVAO() {
    float vertices[] = {
        // Position              Normal             TexCoord
        // Normal faces +Z so the watch screen "emits" light forward
        -0.15f, -0.15f, 0.0f,   0.0f, 0.0f, 1.0f,  0.0f, 0.0f,
         0.15f, -0.15f, 0.0f,   0.0f, 0.0f, 1.0f,  1.0f, 0.0f,
         0.15f,  0.15f, 0.0f,   0.0f, 0.0f, 1.0f,  1.0f, 1.0f,
        -0.15f,  0.15f, 0.0f,   0.0f, 0.0f, 1.0f,  0.0f, 1.0f,
    };

    unsigned int indices[] = { 0, 1, 2, 0, 2, 3 };

    unsigned int VBO, EBO;
    glGenVertexArrays(1, &VAOwatchQuad);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAOwatchQuad);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);

    glBindVertexArray(0);
}

/**
 * Creates a fullscreen quad for 2D rendering (used for FBO)
 * This quad covers the entire normalized device coordinates (-1 to 1)
 * Used to render watch UI elements to the framebuffer texture
 * Vertex format: [Position(2) | TexCoord(2)] = 4 floats per vertex (simpler than 3D)
 */
void createScreenQuadVAO() {
    float vertices[] = {
        // Position(2D)  TexCoord
        -1.0f,  1.0f,    0.0f, 1.0f,   // Top-left
        -1.0f, -1.0f,    0.0f, 0.0f,   // Bottom-left
         1.0f, -1.0f,    1.0f, 0.0f,   // Bottom-right
         1.0f,  1.0f,    1.0f, 1.0f,   // Top-right
    };

    unsigned int indices[] = { 0, 1, 2, 0, 2, 3 };

    unsigned int VBO, EBO;
    glGenVertexArrays(1, &VAOscreenQuad);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAOscreenQuad);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
}

/**
 * Hand uses the same cube geometry, just scaled differently during rendering
 */
void createHandVAO() {
    VAOhand = VAOcube;  // Reuse cube VAO, transform during render
}

// ==================== FRAMEBUFFER SETUP ====================
/*
 * FRAMEBUFFER OBJECT (FBO) EXPLANATION:
 * -------------------------------------
 * By default, OpenGL renders to the screen (default framebuffer).
 * An FBO allows us to render to an off-screen texture instead.
 *
 * Our two-pass rendering process:
 * 1. Bind FBO -> Render watch UI -> Result stored in watchScreenTexture
 * 2. Bind default (0) -> Render 3D scene with watchScreenTexture on watch quad
 *
 * This technique is called "Render-to-Texture" and is commonly used for:
 * - Mirrors, security cameras, portals
 * - Post-processing effects
 * - Shadow mapping
 * - In our case: displaying 2D UI on a 3D surface
 */

/**
 * Creates the framebuffer for rendering the watch screen
 * The FBO has a color attachment (texture) where pixel data is written
 */
void createWatchFramebuffer() {
    // Create and bind the framebuffer
    glGenFramebuffers(1, &watchFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, watchFBO);

    // Create the texture that will receive the rendered image
    glGenTextures(1, &watchScreenTexture);
    glBindTexture(GL_TEXTURE_2D, watchScreenTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WATCH_SCREEN_SIZE, WATCH_SCREEN_SIZE, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, watchScreenTexture, 0);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cout << "Error: Watch framebuffer not complete!" << std::endl;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

// ==================== BUILDING GENERATION ====================
/*
 * Buildings are procedurally generated with random variations in:
 * - Position: slight offset from the road edge
 * - Size: varying width, height, and depth
 * - Color: brownish/beige tones typical of urban buildings
 *
 * Using a fixed seed (42) ensures the same buildings are generated
 * every time the program runs, providing consistent visuals.
 */

/**
 * Generates buildings on both sides of the road
 * Called once at startup to populate the buildings vector
 */
void generateBuildings() {
    buildings.clear();
    srand(42);  // Fixed seed for reproducible results

    for (int side = 0; side < 2; side++) {
        // Left side (side=0) is at negative X, right side (side=1) is at positive X
        float sideX = (side == 0) ? -(ROAD_WIDTH + 5.0f) : (ROAD_WIDTH + 5.0f);

        for (int i = 0; i < NUM_BUILDINGS_PER_SIDE; i++) {
            Building b;

            // Position with slight random offset for natural look
            b.position = glm::vec3(
                sideX + (rand() % 10 - 5) * 0.5f,    // X: road edge + random offset
                0.0f,                                  // Y: ground level
                -10.0f - i * BUILDING_SPACING - (rand() % 10) * 0.5f  // Z: spaced along road
            );

            // Random size within reasonable bounds
            b.scale = glm::vec3(
                4.0f + (rand() % 40) * 0.1f,   // Width: 4-8 meters
                6.0f + (rand() % 100) * 0.1f,  // Height: 6-16 meters
                4.0f + (rand() % 40) * 0.1f    // Depth: 4-8 meters
            );

            // Brownish/beige color with slight variation
            b.color = glm::vec3(
                0.5f + (rand() % 30) * 0.01f,   // Red: 0.5-0.8
                0.45f + (rand() % 30) * 0.01f,  // Green: 0.45-0.75
                0.4f + (rand() % 30) * 0.01f    // Blue: 0.4-0.7
            );

            buildings.push_back(b);
        }
    }
}

// ==================== GLFW CALLBACKS ====================
/*
 * Callbacks are functions called by GLFW when specific events occur.
 * They handle user input asynchronously.
 */

/**
 * Mouse button callback - handles click events for watch UI navigation
 * Only processes clicks when in watch view mode (SPACE pressed)
 */
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
        mouseClicked = true;  // Flag processed in main loop
    }
}

/**
 * Cursor position callback - handles mouse movement
 * In normal mode: controls camera pitch (looking up/down)
 * In watch view mode: tracks cursor for UI interaction
 */
void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
    // Initialize delta tracking on first call
    if (firstMouse) {
        lastMouseX = xpos;
        lastMouseY = ypos;
        firstMouse = false;
    }

    // Calculate movement since last frame
    double xoffset = xpos - lastMouseX;
    double yoffset = lastMouseY - ypos;  // Inverted: moving mouse up = positive yoffset

    lastMouseX = xpos;
    lastMouseY = ypos;

    // Only adjust camera when not in watch view mode
    if (!watchViewMode) {
        float sensitivity = 0.1f;
        cameraBasePitch += (float)yoffset * sensitivity;
        // Clamp pitch to prevent camera flipping
        cameraBasePitch = glm::clamp(cameraBasePitch, -45.0f, 45.0f);
    }

    // Store current position for watch UI hit detection
    mouseX = xpos;
    mouseY = ypos;
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }

    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS) {
        watchViewMode = !watchViewMode;
        //if (watchViewMode) {
        //    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        //   /* glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);*/
        //}
        //else {
        //    // glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        //    /*glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);*/
        //}
    }

    if (key == GLFW_KEY_F1 && action == GLFW_PRESS) {
        depthTestEnabled = !depthTestEnabled;
        std::cout << "Depth testing: " << (depthTestEnabled ? "ON" : "OFF") << std::endl;
    }

    if (key == GLFW_KEY_F2 && action == GLFW_PRESS) {
        faceCullingEnabled = !faceCullingEnabled;
        std::cout << "Face culling: " << (faceCullingEnabled ? "ON" : "OFF") << std::endl;
    }
}

// ==================== UPDATE FUNCTIONS ====================

void updateClock(double currentTime) {
    if (currentTime - lastSecondTime >= 1.0) {
        lastSecondTime = currentTime;
        seconds++;
        if (seconds >= 60) {
            seconds = 0;
            minutes++;
            if (minutes >= 60) {
                minutes = 0;
                hours++;
                if (hours >= 24) {
                    hours = 0;
                }
            }
        }
    }
}

void updateHeartRate(double deltaTime) {
    if (isRunning) {
        targetBpm = (std::min)(targetBpm + 30.0f * (float)deltaTime, 220.0f);
    }
    else {
        targetBpm = (std::max)(targetBpm - 20.0f * (float)deltaTime, 60.0f + (rand() % 20));
    }

    bpm += (targetBpm - bpm) * 2.0f * (float)deltaTime;

    float speed = bpm / 60.0f;
    ekgOffset += speed * (float)deltaTime * 0.5f;
    if (ekgOffset > 1.0f) ekgOffset -= 1.0f;

    float targetScale = 60.0f / bpm;
    ekgScale += (targetScale - ekgScale) * 2.0f * (float)deltaTime;
}

void updateBattery(double currentTime) {
    if (currentTime - lastBatteryDrain >= 10.0 && batteryPercent > 0) {
        lastBatteryDrain = currentTime;
        batteryPercent--;
    }
}

void updateRunning(double deltaTime) {
    if (isRunning && currentScreen == 1) {
        runTime += (float)deltaTime * 8.0f;
        cameraBobOffset = sin(runTime) * 0.05f;
        groundOffset += (float)deltaTime * 8.0f;

        if (groundOffset > GROUND_SEGMENT_LENGTH) {
            groundOffset -= GROUND_SEGMENT_LENGTH;
        }
    }
    else {
        cameraBobOffset *= 0.9f;
    }
}

// ==================== SCREEN DRAWING (2D to FBO) ====================

void drawScreenQuad(unsigned int shader, float x, float y, float w, float h,
    float r, float g, float b, float a,
    unsigned int texture = 0,
    float texScaleX = 1.0f, float texOffsetX = 0.0f) {

    glUseProgram(shader);

    glUniform2f(glGetUniformLocation(shader, "uPos"), x, y);
    glUniform2f(glGetUniformLocation(shader, "uScale"), w, h);
    glUniform4f(glGetUniformLocation(shader, "uColor"), r, g, b, a);
    glUniform1i(glGetUniformLocation(shader, "uUseTexture"), texture != 0 ? 1 : 0);
    glUniform1f(glGetUniformLocation(shader, "uTexScaleX"), texScaleX);
    glUniform1f(glGetUniformLocation(shader, "uTexOffsetX"), texOffsetX);

    if (texture != 0) {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture);
        glUniform1i(glGetUniformLocation(shader, "uTexture"), 0);
    }

    glBindVertexArray(VAOscreenQuad);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

bool isPointInRect(float px, float py, float rx, float ry, float rw, float rh) {
    return px >= rx - rw && px <= rx + rw && py >= ry - rh && py <= ry + rh;
}

void drawClockScreen() {
    char timeStr[16];
    snprintf(timeStr, sizeof(timeStr), "%02d:%02d:%02d", hours, minutes, seconds);

    static unsigned int timeTexture = 0;
    static char lastTimeStr[16] = "";

    if (strcmp(timeStr, lastTimeStr) != 0) {
        if (timeTexture != 0) glDeleteTextures(1, &timeTexture);
        timeTexture = createDigitTexture(timeStr);
        strcpy(lastTimeStr, timeStr);
    }

    drawScreenQuad(screenShader, 0.0f, 0.0f, 0.6f, 0.15f, 1.0f, 1.0f, 1.0f, 1.0f, timeTexture);

    float arrowSize = 0.1f;
    float arrowX = 0.8f;
    drawScreenQuad(screenShader, arrowX, 0.0f, arrowSize, arrowSize, 1.0f, 1.0f, 1.0f, 1.0f, arrowRightTexture);

    if (watchViewMode && mouseClicked) {
        float normMouseX = ((float)mouseX / screenWidth) * 2 - 1;
        float normMouseY = -(((float)mouseY / screenHeight) * 2 - 1);

        if (isPointInRect(normMouseX, normMouseY, arrowX, 0.0f, arrowSize, arrowSize)) {
            currentScreen = 1;
        }
    }
}

void drawHeartRateScreen() {
    float arrowSize = 0.1f;
    float leftArrowX = -0.8f;
    float rightArrowX = 0.8f;

    drawScreenQuad(screenShader, leftArrowX, 0.0f, arrowSize, arrowSize, 1.0f, 1.0f, 1.0f, 1.0f, arrowLeftTexture);
    drawScreenQuad(screenShader, rightArrowX, 0.0f, arrowSize, arrowSize, 1.0f, 1.0f, 1.0f, 1.0f, arrowRightTexture);

    // EKG background
    drawScreenQuad(screenShader, 0.0f, -0.1f, 0.5f, 0.2f, 0.1f, 0.1f, 0.15f, 1.0f);

    // EKG wave
    float numRepeats = 3.0f / ekgScale;
    drawScreenQuad(screenShader, 0.0f, -0.1f, 0.48f, 0.18f, 1.0f, 1.0f, 1.0f, 1.0f,
        ekgTexture, numRepeats, ekgOffset);

    // BPM display
    char bpmStr[16];
    snprintf(bpmStr, sizeof(bpmStr), "%03d", (int)bpm);
    static unsigned int bpmTexture = 0;
    static int lastBpm = 0;

    if ((int)bpm != lastBpm) {
        if (bpmTexture != 0) glDeleteTextures(1, &bpmTexture);
        bpmTexture = createDigitTexture(bpmStr);
        lastBpm = (int)bpm;
    }

    drawScreenQuad(screenShader, 0.0f, 0.25f, 0.2f, 0.1f, 0.0f, 1.0f, 0.4f, 1.0f, bpmTexture);

    // Warning overlay if BPM > 200
    if (bpm > 200) {
        drawScreenQuad(screenShader, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.3f);
    }

    if (watchViewMode && mouseClicked) {
        float normMouseX = ((float)mouseX / screenWidth) * 2 - 1;
        float normMouseY = -(((float)mouseY / screenHeight) * 2 - 1);

        if (isPointInRect(normMouseX, normMouseY, leftArrowX, 0.0f, arrowSize, arrowSize)) {
            currentScreen = 0;
        }
        if (isPointInRect(normMouseX, normMouseY, rightArrowX, 0.0f, arrowSize, arrowSize)) {
            currentScreen = 2;
        }
    }
}

void drawBatteryScreen() {
    float arrowSize = 0.1f;
    float arrowX = -0.8f;

    drawScreenQuad(screenShader, arrowX, 0.0f, arrowSize, arrowSize, 1.0f, 1.0f, 1.0f, 1.0f, arrowLeftTexture);

    // Battery outline
    float battW = 0.3f;
    float battH = 0.15f;
    drawScreenQuad(screenShader, 0.0f, 0.0f, battW, battH, 0.8f, 0.8f, 0.8f, 1.0f);
    drawScreenQuad(screenShader, 0.0f, 0.0f, battW - 0.02f, battH - 0.02f, 0.1f, 0.1f, 0.15f, 1.0f);

    // Battery cap
    drawScreenQuad(screenShader, battW + 0.02f, 0.0f, 0.02f, 0.06f, 0.8f, 0.8f, 0.8f, 1.0f);

    // Battery fill
    float fillPercent = batteryPercent / 100.0f;
    float maxFillW = battW - 0.04f;
    float fillW = maxFillW * fillPercent;
    float fillX = -(maxFillW - fillW);

    float r, g, b;
    if (batteryPercent <= 10) {
        r = 1.0f; g = 0.2f; b = 0.2f;
    }
    else if (batteryPercent <= 20) {
        r = 1.0f; g = 0.8f; b = 0.0f;
    }
    else {
        r = 0.2f; g = 0.9f; b = 0.3f;
    }

    if (batteryPercent > 0) {
        drawScreenQuad(screenShader, fillX, 0.0f, fillW, battH - 0.04f, r, g, b, 1.0f);
    }

    // Percentage display
    char percStr[8];
    snprintf(percStr, sizeof(percStr), "%03d", batteryPercent);
    static unsigned int percTexture = 0;
    static int lastPerc = -1;

    if (batteryPercent != lastPerc) {
        if (percTexture != 0) glDeleteTextures(1, &percTexture);
        percTexture = createDigitTexture(percStr);
        lastPerc = batteryPercent;
    }

    drawScreenQuad(screenShader, 0.0f, 0.3f, 0.15f, 0.08f, 1.0f, 1.0f, 1.0f, 1.0f, percTexture);

    if (watchViewMode && mouseClicked) {
        float normMouseX = ((float)mouseX / screenWidth) * 2 - 1;
        float normMouseY = -(((float)mouseY / screenHeight) * 2 - 1);

        if (isPointInRect(normMouseX, normMouseY, arrowX, 0.0f, arrowSize, arrowSize)) {
            currentScreen = 1;
        }
    }
}

void renderWatchScreen() {
    glBindFramebuffer(GL_FRAMEBUFFER, watchFBO);
    glViewport(0, 0, WATCH_SCREEN_SIZE, WATCH_SCREEN_SIZE);
    glClearColor(0.05f, 0.05f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    switch (currentScreen) {
    case 0:
        drawClockScreen();
        break;
    case 1:
        drawHeartRateScreen();
        break;
    case 2:
        drawBatteryScreen();
        break;
    }

    // Draw cursor when in watch view mode
    if (watchViewMode) {
        float normMouseX = ((float)mouseX / screenWidth) * 2 - 1;
        float normMouseY = -(((float)mouseY / screenHeight) * 2 - 1);
        float cursorSize = 0.04f;
        drawScreenQuad(screenShader, normMouseX, normMouseY, cursorSize, cursorSize, 1.0f, 1.0f, 1.0f, 1.0f, heartCursorTexture);
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

// ==================== 3D SCENE RENDERING ====================

/**
 * Configures the two light sources in the scene
 *
 * LIGHT 1: SUN (uLight)
 * ---------------------
 * - Position: High in the sky (50 units up)
 * - Warm white color (slight yellow/orange tint like sunlight)
 * - Strong ambient (base illumination even in shadows)
 * - Strong diffuse (main lighting contribution)
 * - Strong specular (bright highlights on shiny surfaces)
 *
 * LIGHT 2: WATCH SCREEN (uScreenLight)
 * ------------------------------------
 * - Position: At the watch location (passed as parameter)
 * - Cool blue-white color (like LCD backlight)
 * - WEAK intensity - only noticeable close to the watch
 * - This creates the "screen glow" effect on the hand/nearby objects
 *
 * @param watchPos - Current world position of the watch screen
 */
void setLightUniforms(unsigned int shader, const glm::vec3& watchPos) {
    // ===== LIGHT 1: SUN =====
    // Position high and to the side for dramatic shadows
    setVec3(shader, "uLight.position", glm::vec3(20.0f, 50.0f, 10.0f));
    // Ambient: base light level (shadows aren't completely black)
    setVec3(shader, "uLight.ambient", glm::vec3(0.3f, 0.3f, 0.35f));
    // Diffuse: main light color (warm white, slightly yellow)
    setVec3(shader, "uLight.diffuse", glm::vec3(0.9f, 0.85f, 0.8f));
    // Specular: highlight color (bright white)
    setVec3(shader, "uLight.specular", glm::vec3(1.0f, 0.95f, 0.9f));

    // ===== LIGHT 2: WATCH SCREEN (WEAK EMISSIVE) =====
    // This light follows the watch position
    setVec3(shader, "uScreenLight.position", watchPos);
    // Very weak values - screen glow is subtle
    setVec3(shader, "uScreenLight.ambient", glm::vec3(0.05f, 0.05f, 0.1f));
    setVec3(shader, "uScreenLight.diffuse", glm::vec3(0.1f, 0.15f, 0.2f));  // Slight blue tint
    setVec3(shader, "uScreenLight.specular", glm::vec3(0.05f, 0.05f, 0.1f));
}

/**
 * Sets material properties for Phong lighting
 * Materials define how a surface interacts with light:
 * - ambient: color when not directly lit
 * - diffuse: color under direct lighting (main surface color)
 * - specular: color of shiny highlights
 * - shininess: how focused the specular highlight is (higher = shinier)
 */
void setMaterialUniforms(unsigned int shader, const glm::vec3& ambient, const glm::vec3& diffuse, const glm::vec3& specular, float shininess) {
    setVec3(shader, "uMaterial.ambient", ambient);
    setVec3(shader, "uMaterial.diffuse", diffuse);
    setVec3(shader, "uMaterial.specular", specular);
    setFloat(shader, "uMaterial.shininess", shininess);
}

// ==================== MAIN 3D SCENE RENDERING ====================
/*
 * PHONG LIGHTING MODEL:
 * ---------------------
 * The scene uses Phong shading with two light sources:
 *
 * 1. SUN LIGHT (uLight):
 *    - Position: High above the scene (0, 50, 0)
 *    - Affects all objects in the scene
 *    - Provides main illumination
 *
 * 2. SCREEN LIGHT (uScreenLight):
 *    - Position: At the watch screen location
 *    - Weak intensity (simulates LCD glow)
 *    - Only noticeable in dark areas close to the watch
 *
 * RENDERING ORDER:
 * ----------------
 * 1. Ground segments (tiled for infinite scrolling)
 * 2. Road (slightly elevated to prevent z-fighting)
 * 3. Buildings (wrapped around for infinite running)
 * 4. Hand (attached to watch)
 * 5. Watch frame (bezel around screen)
 * 6. Watch screen (emissive - doesn't receive lighting, only emits)
 */

/**
 * Renders the complete 3D scene
 * @param view - View matrix (camera position/orientation)
 * @param projection - Projection matrix (perspective transformation)
 * @param viewPos - Camera world position (for specular calculation)
 */
void renderScene(const glm::mat4& view, const glm::mat4& projection, const glm::vec3& viewPos) {
    glUseProgram(basicShader);

    // Set camera matrices for vertex transformation
    setMat4(basicShader, "uView", view);
    setMat4(basicShader, "uProjection", projection);
    setVec3(basicShader, "uViewPos", viewPos);  // Needed for specular highlights

    // Calculate watch position for the screen light source
    // The watch acts as a weak light that illuminates nearby objects
    glm::vec3 watchWorldPos;
    if (watchViewMode) {
        // In front of camera when viewing watch
        watchWorldPos = viewPos + glm::vec3(0.0f, 0.0f, -0.5f);
    }
    else {
        // To the right and below when running/walking
        watchWorldPos = viewPos + glm::vec3(0.4f, -0.3f + cameraBobOffset, -0.3f);
    }
    setLightUniforms(basicShader, watchWorldPos);

    // ===== DRAW GROUND SEGMENTS =====
    // Ground uses grass material: moderate ambient, high diffuse, low specular (not shiny)
    setMaterialUniforms(basicShader, glm::vec3(0.3f), glm::vec3(0.8f), glm::vec3(0.1f), 8.0f);
    setInt(basicShader, "uUseTexture", 1);    // Enable texture sampling
    setInt(basicShader, "uIsEmissive", 0);    // Ground receives lighting (not emissive)
    setVec4(basicShader, "uColor", glm::vec4(1.0f));  // White = use texture color directly

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, groundTexture);
    setInt(basicShader, "uTexture", 0);  // Texture unit 0

    // Render multiple ground segments to create infinite scrolling effect
    // groundOffset moves them forward, creating illusion of movement
    glBindVertexArray(VAOground);
    for (int i = 0; i < NUM_GROUND_SEGMENTS; i++) {
        glm::mat4 model = glm::mat4(1.0f);
        // Each segment is placed behind the previous one
        model = glm::translate(model, glm::vec3(0.0f, 0.0f, groundOffset - i * GROUND_SEGMENT_LENGTH));
        setMat4(basicShader, "uModel", model);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    }

    // ===== DRAW ROAD =====
    // Road is rendered slightly above ground (Y=0.01) to prevent z-fighting
    glBindTexture(GL_TEXTURE_2D, roadTexture);
    for (int i = 0; i < NUM_GROUND_SEGMENTS; i++) {
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, glm::vec3(0.0f, 0.01f, groundOffset - i * GROUND_SEGMENT_LENGTH));
        model = glm::scale(model, glm::vec3(ROAD_WIDTH / 100.0f, 1.0f, 1.0f));  // Scale width
        setMat4(basicShader, "uModel", model);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    }

    // ===== DRAW BUILDINGS =====
    // Buildings use slightly shiny material (concrete/plaster look)
    setMaterialUniforms(basicShader, glm::vec3(0.2f), glm::vec3(0.7f), glm::vec3(0.3f), 16.0f);
    glBindTexture(GL_TEXTURE_2D, buildingTexture);
    glBindVertexArray(VAOcube);

    for (const auto& building : buildings) {
        glm::mat4 model = glm::mat4(1.0f);
        glm::vec3 pos = building.position;
        pos.z += groundOffset;  // Move with ground scrolling

        // INFINITE SCROLLING: Wrap buildings when they go too far
        // This creates the illusion of endless buildings along the road
        while (pos.z > 10.0f) pos.z -= NUM_GROUND_SEGMENTS * GROUND_SEGMENT_LENGTH;
        while (pos.z < -NUM_GROUND_SEGMENTS * GROUND_SEGMENT_LENGTH) pos.z += NUM_GROUND_SEGMENTS * GROUND_SEGMENT_LENGTH;

        // Position building: Y is half-height because cube is centered at origin
        model = glm::translate(model, glm::vec3(pos.x, building.scale.y / 2.0f, pos.z));
        model = glm::scale(model, building.scale);
        setMat4(basicShader, "uModel", model);
        setVec4(basicShader, "uColor", glm::vec4(building.color, 1.0f));
        glDrawArrays(GL_TRIANGLES, 0, 36);  // 36 vertices = 6 faces * 2 triangles * 3 vertices
    }

    // ===== DRAW HAND =====
    // Hand uses skin-tone color, no texture, slightly subsurface-scatter look
    setInt(basicShader, "uUseTexture", 0);  // Disable texture, use solid color
    setMaterialUniforms(basicShader, glm::vec3(0.3f), glm::vec3(0.8f, 0.6f, 0.5f), glm::vec3(0.2f), 8.0f);
    setVec4(basicShader, "uColor", glm::vec4(0.9f, 0.75f, 0.65f, 1.0f));  // Skin tone

    glm::mat4 handModel = glm::mat4(1.0f);
    if (watchViewMode) {
        // Hand raised in front of face to look at watch
        handModel = glm::translate(handModel, viewPos + glm::vec3(0.0f, -0.3f, -0.6f));
        handModel = glm::rotate(handModel, glm::radians(-30.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    }
    else {
        // Hand at side with watch, angled naturally
        // cameraBobOffset makes hand bob while running
        handModel = glm::translate(handModel, viewPos + glm::vec3(0.4f, -0.4f + cameraBobOffset * 0.5f, -0.3f));
        handModel = glm::rotate(handModel, glm::radians(-45.0f), glm::vec3(1.0f, 0.0f, 0.0f));
        handModel = glm::rotate(handModel, glm::radians(30.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    }
    // Scale cube to hand/forearm proportions
    handModel = glm::scale(handModel, glm::vec3(0.08f, 0.4f, 0.15f));
    setMat4(basicShader, "uModel", handModel);
    glDrawArrays(GL_TRIANGLES, 0, 36);

    // ===== DRAW WATCH FRAME (BEZEL) =====
    // Dark metallic frame around the screen
    setVec4(basicShader, "uColor", glm::vec4(0.2f, 0.2f, 0.25f, 1.0f));  // Dark gray
    // High specular, high shininess = metallic appearance
    setMaterialUniforms(basicShader, glm::vec3(0.1f), glm::vec3(0.3f), glm::vec3(0.8f), 64.0f);

    glm::mat4 watchFrameModel = glm::mat4(1.0f);
    if (watchViewMode) {
        watchFrameModel = glm::translate(watchFrameModel, viewPos + glm::vec3(0.0f, 0.0f, -0.5f));
    }
    else {
        watchFrameModel = glm::translate(watchFrameModel, viewPos + glm::vec3(0.4f, -0.3f + cameraBobOffset, -0.3f));
        watchFrameModel = glm::rotate(watchFrameModel, glm::radians(-45.0f), glm::vec3(1.0f, 0.0f, 0.0f));
        watchFrameModel = glm::rotate(watchFrameModel, glm::radians(30.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    }
    watchFrameModel = glm::scale(watchFrameModel, glm::vec3(0.35f, 0.35f, 0.03f));
    setMat4(basicShader, "uModel", watchFrameModel);
    glDrawArrays(GL_TRIANGLES, 0, 36);

    // ===== DRAW WATCH SCREEN (EMISSIVE SURFACE) =====
    // The watch screen is EMISSIVE - it emits light rather than receiving it
    // This makes it always fully visible regardless of lighting conditions
    // (like a real LCD/OLED screen that produces its own light)
    setInt(basicShader, "uUseTexture", 1);
    setInt(basicShader, "uIsEmissive", 1);  // KEY: Shader outputs texture color directly, no lighting
    setVec4(basicShader, "uColor", glm::vec4(1.0f));

    // Bind the FBO texture that contains the rendered watch UI
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, watchScreenTexture);  // This is our FBO color attachment

    glm::mat4 watchModel = glm::mat4(1.0f);
    if (watchViewMode) {
        // Directly in front of camera, facing viewer
        watchModel = glm::translate(watchModel, viewPos + glm::vec3(0.0f, 0.0f, -0.48f));
    }
    else {
        // On wrist at side, angled to match hand/frame orientation
        // Z is -0.28 (slightly in front of frame at -0.3)
        watchModel = glm::translate(watchModel, viewPos + glm::vec3(0.4f, -0.3f + cameraBobOffset, -0.28f));
        watchModel = glm::rotate(watchModel, glm::radians(-45.0f), glm::vec3(1.0f, 0.0f, 0.0f));
        watchModel = glm::rotate(watchModel, glm::radians(30.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    }
    setMat4(basicShader, "uModel", watchModel);

    glBindVertexArray(VAOwatchQuad);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

    // Reset emissive flag for next frame
    setInt(basicShader, "uIsEmissive", 0);
    glBindVertexArray(0);
}

void renderStudentInfo() {
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    float infoW = 0.20f;
    float infoH = 0.06f;
    float infoX = 0.79f;
    float infoY = 0.93f;

    drawScreenQuad(screenShader, infoX, infoY, infoW, infoH, 1.0f, 1.0f, 1.0f, 1.0f, studentInfoTexture);

    if (depthTestEnabled) glEnable(GL_DEPTH_TEST);
}

// ==================== MAIN FUNCTION ====================

int main()
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(monitor);
    screenWidth = mode->width;
    screenHeight = mode->height;

    GLFWwindow* window = glfwCreateWindow(screenWidth, screenHeight, "SmartWatch 3D - Nikola Bandulaja SV74/2022", monitor, NULL);
    if (window == NULL) return endProgram("Failed to create window.");
    glfwMakeContextCurrent(window);

    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetKeyCallback(window, key_callback);

    if (glewInit() != GLEW_OK) return endProgram("Failed to initialize GLEW.");

    std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;
    std::cout << "Controls:" << std::endl;
    std::cout << "  Mouse: Look up/down" << std::endl;
    std::cout << "  SPACE: Toggle watch view mode" << std::endl;
    std::cout << "  D (hold): Simulate running (on heart rate screen)" << std::endl;
    std::cout << "  F1: Toggle depth testing" << std::endl;
    std::cout << "  F2: Toggle face culling" << std::endl;
    std::cout << "  ESC: Exit" << std::endl;

    // Create shaders
    basicShader = createShader("basic.vert", "basic.frag");
    screenShader = createShader("screen.vert", "screen.frag");

    // Create VAOs
    createGroundVAO();
    createCubeVAO();
    createWatchQuadVAO();
    createScreenQuadVAO();
    createHandVAO();

    // Create textures
    groundTexture = createGroundTexture();
    roadTexture = createRoadTexture();
    buildingTexture = createBuildingTexture();
    ekgTexture = createEKGTexture();
    arrowRightTexture = createArrowTexture(true);
    arrowLeftTexture = createArrowTexture(false);
    heartCursorTexture = createHeartTexture();
    studentInfoTexture = createStudentInfoTexture();

    // Create framebuffer for watch screen
    createWatchFramebuffer();

    // Generate buildings
    generateBuildings();

    // Initialize timing
    srand((unsigned)time(NULL));
    double lastTime = glfwGetTime();
    lastSecondTime = lastTime;
    lastBatteryDrain = lastTime;

    glClearColor(0.4f, 0.6f, 0.9f, 1.0f);

    while (!glfwWindowShouldClose(window))
    {
        double currentTime = glfwGetTime();
        double deltaTime = currentTime - lastTime;

        // Frame limiter
        if (deltaTime < TARGET_FRAME_TIME) {
            double sleepTime = TARGET_FRAME_TIME - deltaTime;
            std::this_thread::sleep_for(std::chrono::microseconds((int)(sleepTime * 1000000)));
            currentTime = glfwGetTime();
            deltaTime = currentTime - lastTime;
        }
        lastTime = currentTime;

        // Check running state
        isRunning = (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) && (currentScreen == 1);

        // Update state
        updateClock(currentTime);
        updateHeartRate(deltaTime);
        updateBattery(currentTime);
        updateRunning(deltaTime);

        // Update camera
        cameraPitch = cameraBasePitch + cameraBobOffset * 100.0f;
        cameraPos.y = 1.6f + cameraBobOffset;

        // Render watch screen to FBO
        renderWatchScreen();

        // Render 3D scene
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, screenWidth, screenHeight);

        if (depthTestEnabled) glEnable(GL_DEPTH_TEST);
        else glDisable(GL_DEPTH_TEST);

        if (faceCullingEnabled) {
            glEnable(GL_CULL_FACE);
            glFrontFace(GL_CCW);
            glCullFace(GL_BACK);
        }
        else {
            glDisable(GL_CULL_FACE);
        }

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Camera matrices
        glm::vec3 cameraFront;
        cameraFront.x = cos(glm::radians(cameraYaw)) * cos(glm::radians(cameraPitch));
        cameraFront.y = sin(glm::radians(cameraPitch));
        cameraFront.z = sin(glm::radians(cameraYaw)) * cos(glm::radians(cameraPitch));
        cameraFront = glm::normalize(cameraFront);

        glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 projection = glm::perspective(glm::radians(60.0f), (float)screenWidth / (float)screenHeight, 0.1f, 200.0f);

        renderScene(view, projection, cameraPos);

        // Render student info overlay
        glViewport(0, 0, screenWidth, screenHeight);
        renderStudentInfo();

        mouseClicked = false;

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup
    glDeleteTextures(1, &groundTexture);
    glDeleteTextures(1, &roadTexture);
    glDeleteTextures(1, &buildingTexture);
    glDeleteTextures(1, &ekgTexture);
    glDeleteTextures(1, &arrowRightTexture);
    glDeleteTextures(1, &arrowLeftTexture);
    glDeleteTextures(1, &heartCursorTexture);
    glDeleteTextures(1, &studentInfoTexture);
    glDeleteTextures(1, &watchScreenTexture);

    glDeleteFramebuffers(1, &watchFBO);

    glDeleteVertexArrays(1, &VAOground);
    glDeleteVertexArrays(1, &VAOcube);
    glDeleteVertexArrays(1, &VAOwatchQuad);
    glDeleteVertexArrays(1, &VAOscreenQuad);

    glDeleteProgram(basicShader);
    glDeleteProgram(screenShader);

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
