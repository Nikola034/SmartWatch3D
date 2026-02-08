#version 330 core

// Light source properties
struct Light {
    vec3 position;   // World-space position of the light
    vec3 ambient;    // Ambient light color/intensity
    vec3 diffuse;    // Diffuse light color/intensity
    vec3 specular;   // Specular light color/intensity
};

// Material surface properties
struct Material {
    vec3 ambient;    // How much ambient light the surface reflects
    vec3 diffuse;    // How much diffuse light the surface reflects (main color)
    vec3 specular;   // How much specular light the surface reflects (highlight color)
    float shininess; // How focused the specular highlight is (higher = smaller, sharper)
};

// Inputs from vertex shader (interpolated across triangle)
in vec3 fragPos;    // Fragment position in world space
in vec3 normal;     // Surface normal in world space
in vec2 texCoord;   // Texture coordinates

// Output color
out vec4 outColor;

// Uniforms set from CPU
uniform Light uLight;         // Main sun light
uniform Light uScreenLight;   // Watch screen light (weak)
uniform Material uMaterial;   // Current surface material
uniform vec3 uViewPos;        // Camera position (for specular calculation)
uniform sampler2D uTexture;   // Texture sampler
uniform int uUseTexture;      // 1 = sample texture, 0 = use solid color
uniform vec4 uColor;          // Solid color (or texture multiplier)
uniform int uIsEmissive;      // 1 = emit light, 0 = receive light


vec3 calculateLight(Light light, vec3 norm, vec3 viewDir, vec3 baseColor) {
    // === AMBIENT ===
    // Always present, simulates indirect light
    vec3 ambient = light.ambient * uMaterial.ambient * baseColor;

    // === DIFFUSE ===
    // Direction from fragment to light source
    vec3 lightDir = normalize(light.position - fragPos);
    // Dot product gives cos(angle) - clamped to 0 for back-facing surfaces
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = light.diffuse * diff * uMaterial.diffuse * baseColor;

    // === SPECULAR ===
    // Reflect light direction around normal
    vec3 reflectDir = reflect(-lightDir, norm);
    // Dot product with view direction, raised to shininess power
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), uMaterial.shininess);
    vec3 specular = light.specular * spec * uMaterial.specular;

    return ambient + diffuse + specular;
}

void main()
{
    // Normalize the interpolated normal (interpolation can denormalize it)
    vec3 norm = normalize(normal);

    // Direction from fragment to camera (for specular)
    vec3 viewDir = normalize(uViewPos - fragPos);

    // Determine base color from texture or solid color
    vec3 baseColor;
    if (uUseTexture == 1) {
        baseColor = texture(uTexture, texCoord).rgb;
    } else {
        baseColor = uColor.rgb;
    }

    // Check if this is an emissive surface (like the watch screen)
    if (uIsEmissive == 1) {
        // EMISSIVE: Object produces its own light
        // Output texture color directly, slightly brightened (1.2x)
        // No lighting calculations - screen is always fully visible
        outColor = vec4(baseColor * 1.2, uColor.a);
    } else {
        // NORMAL: Object receives light from light sources

        // Calculate contribution from main sun light
        vec3 result = calculateLight(uLight, norm, viewDir, baseColor);

        // Add contribution from watch screen light (scaled to 30%)
        // This creates the subtle screen glow effect on nearby surfaces
        result += calculateLight(uScreenLight, norm, viewDir, baseColor) * 0.3;

        outColor = vec4(result, uColor.a);
    }
}
