Shader "Unlit/half"
{
    Properties
    {
        _Albedo ("Albedo", Color) = (1, 1, 1, 1)
        _Metallic ("Metallic", Range(0, 1)) = 1
        _Smoothness ("Smoothness", Range(0, 1)) = 1
        _LightColor ("Light Color", Color) = (1, 1, 1, 1)
        _LightDir ("Light Dir", Vector) = (1, 0, 0, 0)
        _IndirectDiffuse("Indirect Diffuse", Color) = (1, 1, 1, 1)
        _IndirectSpecular("Indirect Specular", Color) = (1, 1, 1, 1)
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

            #define _type4 half4
            #define _type3 half3
            #define _type2 half2
            #define _type half


            struct appdata
            {
                _type4 vertex : POSITION;
                _type3 normal : NORMAL;
            };

            struct v2f
            {
                _type4 vertex : SV_POSITION;
                _type3 normal : TEXCOORD1;
                _type3 posWorld : TEXCOORD4;
            };

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.normal = UnityObjectToWorldNormal(v.normal);
                o.posWorld = mul(unity_ObjectToWorld, v.vertex);
                return o;
            }

            inline _type OneMinusReflectivityFromMetallic(_type metallic)
            {
                // We'll need oneMinusReflectivity, so
                //   1-reflectivity = 1-lerp(dielectricSpec, 1, metallic) = lerp(1-dielectricSpec, 0, metallic)
                // store (1-dielectricSpec) in unity_ColorSpaceDielectricSpec.a, then
                //   1-reflectivity = lerp(alpha, 0, metallic) = alpha + metallic*(0 - alpha) =
                //                  = alpha - metallic * alpha
                _type oneMinusDielectricSpec = unity_ColorSpaceDielectricSpec.a;
                return oneMinusDielectricSpec - metallic * oneMinusDielectricSpec;
            }

            inline _type3 DiffuseAndSpecularFromMetallic(_type3 albedo, _type metallic, out _type3 specColor, out _type oneMinusReflectivity)
            {
                specColor = lerp(unity_ColorSpaceDielectricSpec.rgb, albedo, metallic);
                oneMinusReflectivity = OneMinusReflectivityFromMetallic(metallic);
                return albedo * oneMinusReflectivity;
            }

            _type PerceptualRoughnessToRoughness(_type perceptualRoughness)
            {
                return perceptualRoughness * perceptualRoughness;
            }

            _type SmoothnessToPerceptualRoughness(_type smoothness)
            {
                return (1 - smoothness);
            }

            inline _type3 Unity_SafeNormalize(_type3 inVec)
            {
                _type dp3 = max(0.001f, dot(inVec, inVec));
                return inVec * rsqrt(dp3);
            }

            inline _type Pow4(_type x)
            {
                return x * x * x * x;
            }

            inline _type2 Pow4(_type2 x)
            {
                return x * x * x * x;
            }

            inline _type3 Pow4(_type3 x)
            {
                return x * x * x * x;
            }

            inline _type4 Pow4(_type4 x)
            {
                return x * x * x * x;
            }

            // Pow5 uses the same amount of instructions as generic pow(), but has 2 advantages:
            // 1) better instruction pipelining
            // 2) no need to worry about NaNs
            inline _type Pow5(_type x)
            {
                return x * x * x * x * x;
            }

            inline _type2 Pow5(_type2 x)
            {
                return x * x * x * x * x;
            }

            inline _type3 Pow5(_type3 x)
            {
                return x * x * x * x * x;
            }

            inline _type4 Pow5(_type4 x)
            {
                return x * x * x * x * x;
            }

            inline _type3 FresnelLerp(_type3 F0, _type3 F90, _type cosA)
            {
                _type t = Pow5(1 - cosA);   // ala Schlick interpoliation
                return lerp(F0, F90, t);
            }
            // approximage Schlick with ^4 instead of ^5
            inline _type3 FresnelLerpFast(_type3 F0, _type3 F90, _type cosA)
            {
                _type t = Pow4(1 - cosA);
                return lerp(F0, F90, t);
            }

            // BlinnPhong normalized as normal distribution function (NDF)
            // for use in micro-facet model: spec=D*G*F
            // eq. 19 in https://dl.dropboxusercontent.com/u/55891920/papers/mm_brdf.pdf
            inline _type NDFBlinnPhongNormalizedTerm(_type NdotH, _type n)
            {
                // norm = (n+2)/(2*pi)
                _type normTerm = (n + 2.0) * (0.5 / UNITY_PI);

                _type specTerm = pow(NdotH, n);
                return specTerm * normTerm;
            }

            #define UNITY_BRDF_GGX 1

            _type4 BRDF2_Unity_PBS (_type3 diffColor, _type3 specColor, _type oneMinusReflectivity, _type smoothness,
                _type3 normal, _type3 viewDir,
                UnityLight light, UnityIndirect gi)
            {
                _type3 halfDir = Unity_SafeNormalize (_type3(light.dir) + viewDir);

                _type nl = saturate(dot(normal, light.dir));
                _type nh = saturate(dot(normal, halfDir));
                _type nv = saturate(dot(normal, viewDir));
                _type lh = saturate(dot(light.dir, halfDir));

                // Specular term
                _type perceptualRoughness = SmoothnessToPerceptualRoughness (smoothness);
                _type roughness = PerceptualRoughnessToRoughness(perceptualRoughness);

            #if UNITY_BRDF_GGX

                // GGX Distribution multiplied by combined approximation of Visibility and Fresnel
                // See "Optimizing PBR for Mobile" from Siggraph 2015 moving mobile graphics course
                // https://community.arm.com/events/1155
                _type a = roughness;
                _type a2 = a*a;

                _type d = nh * nh * (a2 - 1.f) + 1.00001f;
            #ifdef UNITY_COLORSPACE_GAMMA
                // Tighter approximation for Gamma only rendering mode!
                // DVF = sqrt(DVF);
                // DVF = (a * sqrt(.25)) / (max(sqrt(0.1), lh)*sqrt(roughness + .5) * d);
                _type specularTerm = a / (max(0.32f, lh) * (1.5f + roughness) * d);
            #else
                _type specularTerm = a2 / (max(0.1f, lh*lh) * (roughness + 0.5f) * (d * d) * 4);
            #endif

                // on mobiles (where _type actually means something) denominator have risk of overflow
                // clamp below was added specifically to "fix" that, but dx compiler (we convert bytecode to metal/gles)
                // sees that specularTerm have only non-negative terms, so it skips max(0,..) in clamp (leaving only min(100,...))
            #if defined (SHADER_API_MOBILE)
                specularTerm = specularTerm - 1e-4f;
            #endif

            #else

                // Legacy
                _type specularPower = PerceptualRoughnessToSpecPower(perceptualRoughness);
                // Modified with approximate Visibility function that takes roughness into account
                // Original ((n+1)*N.H^n) / (8*Pi * L.H^3) didn't take into account roughness
                // and produced extremely bright specular at grazing angles

                _type invV = lh * lh * smoothness + perceptualRoughness * perceptualRoughness; // approx ModifiedKelemenVisibilityTerm(lh, perceptualRoughness);
                _type invF = lh;

                _type specularTerm = ((specularPower + 1) * pow (nh, specularPower)) / (8 * invV * invF + 1e-4h);

            #ifdef UNITY_COLORSPACE_GAMMA
                specularTerm = sqrt(max(1e-4f, specularTerm));
            #endif

            #endif

            #if defined (SHADER_API_MOBILE)
                specularTerm = clamp(specularTerm, 0.0, 100.0); // Prevent FP16 overflow on mobiles
            #endif
            #if defined(_SPECULARHIGHLIGHTS_OFF)
                specularTerm = 0.0;
            #endif

                // surfaceReduction = Int D(NdotH) * NdotH * Id(NdotL>0) dH = 1/(realRoughness^2+1)

                // 1-0.28*x^3 as approximation for (1/(x^4+1))^(1/2.2) on the domain [0;1]
                // 1-x^3*(0.6-0.08*x)   approximation for 1/(x^4+1)
            #ifdef UNITY_COLORSPACE_GAMMA
                _type surfaceReduction = 0.28;
            #else
                _type surfaceReduction = (0.6-0.08*perceptualRoughness);
            #endif

                surfaceReduction = 1.0 - roughness*perceptualRoughness*surfaceReduction;

                _type grazingTerm = saturate(smoothness + (1-oneMinusReflectivity));
                _type3 color =   (diffColor + specularTerm * specColor) * light.color * nl
                                + gi.diffuse * diffColor
                                + surfaceReduction * gi.specular * FresnelLerpFast (specColor, grazingTerm, nv);

                return _type4(color, 1);
            }

            _type4 _Albedo;
            _type _Metallic;
            _type _Smoothness;
            _type3 _LightColor;
            _type3 _LightDir;
            _type3 _IndirectDiffuse;
            _type3 _IndirectSpecular;

            _type4 frag(v2f i) : SV_Target
            {
                _type3 diffuse;
                _type3 specular;
                _type oneMinusReflectivity;
                diffuse = DiffuseAndSpecularFromMetallic(_Albedo, _Metallic, specular, oneMinusReflectivity);

                _type3 viewDir = normalize(UnityWorldSpaceViewDir(i.posWorld));

                UnityLight light;
                light.color = _LightColor;
                light.dir = _LightDir;

                UnityIndirect indirect;
                indirect.diffuse = _IndirectDiffuse;
                indirect.specular = _IndirectSpecular;

                _type4 final = BRDF2_Unity_PBS(diffuse, specular, oneMinusReflectivity, _Smoothness, normalize(i.normal), viewDir, light, indirect);
                return final;
            }
            ENDCG
        }
    }
}
