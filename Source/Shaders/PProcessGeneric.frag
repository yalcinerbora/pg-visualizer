#version 430
/*
	**Post Process Generic Shader**

	File Name	: PProcessGeneric.frag
	Author		: Bora Yalciner
	Description	:

		Pass Trough Shader
*/

// Definitions
#define IN_UV layout(location = 0)
#define OUT_COLOR layout(location = 0)

#define T_IN_COLOR layout(binding = 0)

// Input
in IN_UV vec2 fUV;

// Output
out OUT_COLOR vec4 fboColor;

// Textures
uniform T_IN_COLOR sampler2D gColor;

void main(void)
{
	fboColor = vec4(texture(gColor, fUV).xyz, 1.0f);
}