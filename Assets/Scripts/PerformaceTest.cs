using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

[RequireComponent(typeof(Camera))]
public class PerformaceTest : MonoBehaviour
{
    public Shader BRDF1;
    public Shader BRDF2;
    public Shader BRDF3;
    public Shader BlinnPhong;
    public Shader Lambert;
    public Shader Float;
    public Shader Half;
    public Shader Fixed;

    public Color albedo = new Color(1, 1, 1);
    [Range(0, 1)]
    public float metallic;
    [Range(0, 1)]
    public float smoothness;
    public Color specularColor = new Color(1, 1, 1);
    [Range(0, 1)]
    public float specular;
    [Range(0, 1)]
    public float gloss;
    public Color lightColor = new Color(1, 1, 1);
    public Vector3 lightDir = new Vector3(0, -1, 0);
    public Color indirectDiffuse = new Color(0.1f, 0.1f, 0.1f);
    public Color indirectSpecular = new Color(0.1f, 0.1f, 0.1f);
    public Texture2D nhx;

    [Min(0)]
    public int count = 500;
    public int fpsThreshod;
    bool testing = false; 

    Material mat;
    RenderTexture rt1;
    RenderTexture rt2;

    float deltaTime = 0.0f;
    float fps = 0;
    float lastcount;
    float adaptTime = 0;

    GUIStyle mStyle;

    string gpudevice;

    // Start is called before the first frame update
    void Start()
    {
        rt1 = new RenderTexture(1024, 1024, 0, RenderTextureFormat.ARGB32);
        rt2 = new RenderTexture(1024, 1024, 0, RenderTextureFormat.ARGB32);

        OnValidate();

        mStyle = new GUIStyle();
        mStyle.alignment = TextAnchor.UpperLeft;
        mStyle.normal.background = null;
        mStyle.fontSize = 25;
        mStyle.normal.textColor = new Color(0f, 1f, 0f, 1.0f);

        gpudevice = SystemInfo.graphicsDeviceName;
    }

    private void OnValidate()
    {
        if (!mat)
            return;

        mat.SetColor("_Albedo", albedo);
        mat.SetFloat("_Metallic", metallic);
        mat.SetFloat("_Smoothness", smoothness);
        mat.SetColor("_SpecularColor", specularColor);
        mat.SetFloat("_Specular", specular);
        mat.SetFloat("_Gloss", gloss);
        mat.SetColor("_LightColor", lightColor);
        mat.SetVector("_LightDir", lightDir.normalized);
        mat.SetColor("_IndirectDiffuse", indirectDiffuse);
        mat.SetColor("_IndirectSpecular", indirectSpecular);
        mat.SetTexture("_NHX", nhx);
    }

    private void OnDestroy()
    {
        if (mat)
            Destroy(mat);
        if (rt1)
            Destroy(rt1);
        if (rt2)
            Destroy(rt2);
    }

    private void OnRenderImage(RenderTexture source, RenderTexture destination)
    {
        for (int i = 0; i < count && testing; ++i)
            Graphics.Blit(rt1, rt2, mat);
        Graphics.Blit(source, destination);
    }

    private void OnGUI()
    {
        int w = Screen.width;
        int h = Screen.height;
        Rect rect = new Rect(100, 100, 100, 100);
        
        string text = string.Format("   {0:0.} FPS", fps);
        GUI.Label(rect, text, mStyle);

        Rect rect2 = new Rect(100, 200, 100, 100);
        string text2 = string.Format("   {0:0.} Count", count);
        GUI.Label(rect2, text2, mStyle);

        if (!testing)
        {
            if (GUI.Button(new Rect(100, 300, 100, 100), "30"))
            {
                fpsThreshod = 30;
            }
            if (GUI.Button(new Rect(100, 400, 100, 100), "15"))
            {
                fpsThreshod = 15;
            }

            GUI.Label(new Rect(100, 500, 100, 100), gpudevice);

            if (GUI.Button(new Rect(300, 100, 100, 100), "BRDF1"))
            {
                mat = new Material(BRDF1);
            }
            if (GUI.Button(new Rect(300, 200, 100, 100), "BRDF2"))
            {
                mat = new Material(BRDF2);
            }
            if (GUI.Button(new Rect(300, 300, 100, 100), "BRDF3"))
            {
                mat = new Material(BRDF3);
            }
            if (GUI.Button(new Rect(400, 100, 100, 100), "Blinn Phong"))
            {
                mat = new Material(BlinnPhong);
            }
            if (GUI.Button(new Rect(400, 200, 100, 100), "Lambert"))
            {
                mat = new Material(Lambert);
            }

            if (GUI.Button(new Rect(500, 100, 100, 100), "Float"))
            {
                mat = new Material(Float);
            }
            if (GUI.Button(new Rect(500, 200, 100, 100), "Half"))
            {
                mat = new Material(Half);
            }
            if (GUI.Button(new Rect(500, 300, 100, 100), "Fixed"))
            {
                mat = new Material(Fixed);
            }

            testing = mat != null;
            if (testing)
            {
                OnValidate();
            }
        }
        else
        {
            if (GUI.Button(new Rect(300, 100, 100, 100), "Back"))
            {
                testing = false;
                Destroy(mat);
            }
            if (GUI.Button(new Rect(300, 200, 100, 100), "Count X 2"))
            {
                count *= 2;
            }
        }
    }

    private void Update()
    {
        deltaTime += (Time.deltaTime - deltaTime) * 0.1f;
        fps = 1.0f / deltaTime;
        if (testing)
        {
            adaptTime += Time.deltaTime;
            if (adaptTime > 0.2)
            {
                float c = fps - fpsThreshod;
                float newcount = count + c;
                newcount = Mathf.Max(newcount, 1);
                newcount = (lastcount + newcount) / 2;
                lastcount = count;
                if (fps < fpsThreshod - 1)
                    count = Mathf.FloorToInt(newcount);
                else
                    count = Mathf.CeilToInt(newcount);
                adaptTime = 0;
            }
        }
    }
}
