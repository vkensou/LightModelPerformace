Shader "Unlit/Lambert"
{
    Properties
    {
        _Albedo ("Albedo", Color) = (1, 1, 1, 1)
        _Metallic ("Metallic", Range(0, 1)) = 1
        _Smoothness ("Smoothness", Range(0, 1)) = 1
        _SpecularColor ("Specular Color", Color) = (1, 1, 1, 1)
        _Specular("Specular", Range(0, 1)) = 1
        _Gloss("Gloss", Range(0, 1)) = 1
        _LightColor ("Light Color", Color) = (1, 1, 1, 1)
        _LightDir ("Light Dir", Vector) = (1, 0, 0, 0)
        _IndirectDiffuse("Indirect Diffuse", Color) = (1, 1, 1, 1)
        _IndirectSpecular("Indirect Specular", Color) = (1, 1, 1, 1)
        _NHX("NHX", 2D) = "white"
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            // make fog work
            #pragma multi_compile_fog

            #include "UnityCG.cginc"
            #include "UnityLightingCommon.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float3 normal : NORMAL;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
                float3 normal : TEXCOORD1;
                float3 posWorld : TEXCOORD4;
            };

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.normal = UnityObjectToWorldNormal(v.normal);
                o.posWorld = mul(unity_ObjectToWorld, v.vertex);
                o.uv = v.uv;
                return o;
            }

            inline half OneMinusReflectivityFromMetallic(half metallic)
            {
                // We'll need oneMinusReflectivity, so
                //   1-reflectivity = 1-lerp(dielectricSpec, 1, metallic) = lerp(1-dielectricSpec, 0, metallic)
                // store (1-dielectricSpec) in unity_ColorSpaceDielectricSpec.a, then
                //   1-reflectivity = lerp(alpha, 0, metallic) = alpha + metallic*(0 - alpha) =
                //                  = alpha - metallic * alpha
                half oneMinusDielectricSpec = unity_ColorSpaceDielectricSpec.a;
                return oneMinusDielectricSpec - metallic * oneMinusDielectricSpec;
            }

            inline half3 DiffuseAndSpecularFromMetallic(half3 albedo, half metallic, out half3 specColor, out half oneMinusReflectivity)
            {
                specColor = lerp(unity_ColorSpaceDielectricSpec.rgb, albedo, metallic);
                oneMinusReflectivity = OneMinusReflectivityFromMetallic(metallic);
                return albedo * oneMinusReflectivity;
            }

            float PerceptualRoughnessToRoughness(float perceptualRoughness)
            {
                return perceptualRoughness * perceptualRoughness;
            }

            half RoughnessToPerceptualRoughness(half roughness)
            {
                return sqrt(roughness);
            }

            // Smoothness is the user facing name
            // it should be perceptualSmoothness but we don't want the user to have to deal with this name
            half SmoothnessToRoughness(half smoothness)
            {
                return (1 - smoothness) * (1 - smoothness);
            }

            float SmoothnessToPerceptualRoughness(float smoothness)
            {
                return (1 - smoothness);
            }

            inline float3 Unity_SafeNormalize(float3 inVec)
            {
                float dp3 = max(0.001f, dot(inVec, inVec));
                return inVec * rsqrt(dp3);
            }

            inline half Pow4(half x)
            {
                return x * x * x * x;
            }

            inline float2 Pow4(float2 x)
            {
                return x * x * x * x;
            }

            inline half3 Pow4(half3 x)
            {
                return x * x * x * x;
            }

            inline half4 Pow4(half4 x)
            {
                return x * x * x * x;
            }

            // Pow5 uses the same amount of instructions as generic pow(), but has 2 advantages:
            // 1) better instruction pipelining
            // 2) no need to worry about NaNs
            inline half Pow5(half x)
            {
                return x * x * x * x * x;
            }

            inline half2 Pow5(half2 x)
            {
                return x * x * x * x * x;
            }

            inline half3 Pow5(half3 x)
            {
                return x * x * x * x * x;
            }

            inline half4 Pow5(half4 x)
            {
                return x * x * x * x * x;
            }

            inline half3 FresnelTerm(half3 F0, half cosA)
            {
                half t = Pow5(1 - cosA);   // ala Schlick interpoliation
                return F0 + (1 - F0) * t;
            }
            inline half3 FresnelLerp(half3 F0, half3 F90, half cosA)
            {
                half t = Pow5(1 - cosA);   // ala Schlick interpoliation
                return lerp(F0, F90, t);
            }
            // approximage Schlick with ^4 instead of ^5
            inline half3 FresnelLerpFast(half3 F0, half3 F90, half cosA)
            {
                half t = Pow4(1 - cosA);
                return lerp(F0, F90, t);
            }

            half DisneyDiffuse(half NdotV, half NdotL, half LdotH, half perceptualRoughness)
            {
                half fd90 = 0.5 + 2 * LdotH * LdotH * perceptualRoughness;
                // Two schlick fresnel term
                half lightScatter = (1 + (fd90 - 1) * Pow5(1 - NdotL));
                half viewScatter = (1 + (fd90 - 1) * Pow5(1 - NdotV));

                return lightScatter * viewScatter;
            }

            inline half SmithVisibilityTerm(half NdotL, half NdotV, half k)
            {
                half gL = NdotL * (1 - k) + k;
                half gV = NdotV * (1 - k) + k;
                return 1.0 / (gL * gV + 1e-5f); // This function is not intended to be running on Mobile,
                                                // therefore epsilon is smaller than can be represented by half
            }

            // Smith-Schlick derived for Beckmann
            inline half SmithBeckmannVisibilityTerm(half NdotL, half NdotV, half roughness)
            {
                half c = 0.797884560802865h; // c = sqrt(2 / Pi)
                half k = roughness * c;
                return SmithVisibilityTerm(NdotL, NdotV, k) * 0.25f; // * 0.25 is the 1/4 of the visibility term
            }

            // Ref: http://jcgt.org/published/0003/02/03/paper.pdf
            inline float SmithJointGGXVisibilityTerm(float NdotL, float NdotV, float roughness)
            {
#if 0
                // Original formulation:
                //  lambda_v    = (-1 + sqrt(a2 * (1 - NdotL2) / NdotL2 + 1)) * 0.5f;
                //  lambda_l    = (-1 + sqrt(a2 * (1 - NdotV2) / NdotV2 + 1)) * 0.5f;
                //  G           = 1 / (1 + lambda_v + lambda_l);

                // Reorder code to be more optimal
                half a = roughness;
                half a2 = a * a;

                half lambdaV = NdotL * sqrt((-NdotV * a2 + NdotV) * NdotV + a2);
                half lambdaL = NdotV * sqrt((-NdotL * a2 + NdotL) * NdotL + a2);

                // Simplify visibility term: (2.0f * NdotL * NdotV) /  ((4.0f * NdotL * NdotV) * (lambda_v + lambda_l + 1e-5f));
                return 0.5f / (lambdaV + lambdaL + 1e-5f);  // This function is not intended to be running on Mobile,
                                                            // therefore epsilon is smaller than can be represented by half
#else
                // Approximation of the above formulation (simplify the sqrt, not mathematically correct but close enough)
                float a = roughness;
                float lambdaV = NdotL * (NdotV * (1 - a) + a);
                float lambdaL = NdotV * (NdotL * (1 - a) + a);

#if defined(SHADER_API_SWITCH)
                return 0.5f / (lambdaV + lambdaL + 1e-4f); // work-around against hlslcc rounding error
#else
                return 0.5f / (lambdaV + lambdaL + 1e-5f);
#endif

#endif
            }

            inline float GGXTerm(float NdotH, float roughness)
            {
                float a2 = roughness * roughness;
                float d = (NdotH * a2 - NdotH) * NdotH + 1.0f; // 2 mad
                return UNITY_INV_PI * a2 / (d * d + 1e-7f); // This function is not intended to be running on Mobile,
                                                        // therefore epsilon is smaller than what can be represented by half
            }

            inline half PerceptualRoughnessToSpecPower(half perceptualRoughness)
            {
                half m = PerceptualRoughnessToRoughness(perceptualRoughness);   // m is the true academic roughness.
                half sq = max(1e-4f, m * m);
                half n = (2.0 / sq) - 2.0;                          // https://dl.dropboxusercontent.com/u/55891920/papers/mm_brdf.pdf
                n = max(n, 1e-4f);                                  // prevent possible cases of pow(0,0), which could happen when roughness is 1.0 and NdotH is zero
                return n;
            }

            inline half4 UnityBlinnPhongLight(half3 diffColor, half3 specColor, half specular, half gloss, float3 normal, half3 viewDir, UnityLight light, UnityIndirect indirect)
            {
                half diff = max(0, dot(normal, light.dir));

                half4 c;
                c.rgb = diffColor * light.color * diff;
                c.rgb += diffColor * indirect.diffuse;
                c.a = 1;

                return c;
            }

            half4 _Albedo;
            half _Metallic;
            half _Smoothness;
            half4 _SpecularColor;
            half _Specular;
            half _Gloss;
            half3 _LightColor;
            half3 _LightDir;
            half3 _IndirectDiffuse;
            half3 _IndirectSpecular;

            half4 frag(v2f i) : SV_Target
            {
                half3 viewDir = normalize(UnityWorldSpaceViewDir(i.posWorld));

                UnityLight light;
                light.color = _LightColor;
                light.dir = _LightDir;

                UnityIndirect indirect;
                indirect.diffuse = _IndirectDiffuse;
                indirect.specular = _IndirectSpecular;

                half4 final = UnityBlinnPhongLight(_Albedo, _SpecularColor, _Specular, _Gloss, normalize(i.normal), viewDir, light, indirect);
                return final;
            }
            ENDCG
        }
    }
}
